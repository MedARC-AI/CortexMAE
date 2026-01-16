# This source code is licensed under the Apache License, Version 2.0
#
# References:
# deit: https://github.com/facebookresearch/deit/blob/main/main.py
# capi: https://github.com/facebookresearch/capi/blob/main/train_capi.py

import argparse
import datetime
import json
import math
import random
import time
from functools import partial
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import datasets as hfds
import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from webdataset import WebLoader

import flat_mae.data as flat_data
import flat_mae.utils as ut
import flat_mae.models_mae as models_mae
import flat_mae.masking as masking
import flat_mae.transforms as transforms
import flat_mae.visualization as vis

# quiet noisy hf progress bars when downloading data
hfds.disable_progress_bars()

DEFAULT_CONFIG = Path(__file__).parent / "config/default_pretrain.yaml"

MODELS_DICT = models_mae.__dict__


def main(args: DictConfig):
    # setup
    ut.init_distributed_mode(args)
    global_rank = ut.get_rank()
    is_master = global_rank == 0
    world_size = ut.get_world_size()
    device = torch.device(args.device)
    ut.random_seed(args.seed, rank=global_rank)

    if args.name and not args.output_dir.endswith(args.name):
        args.output_dir = f"{args.output_dir}/{args.name}"
    output_dir = Path(args.output_dir)

    # override config
    # historically config has been too general allowing options that actually can't
    # change. here we override which preserves backward compatibility of the config and
    # model loading.
    args.in_chans = 1
    args.input_space = args.get("input_space", "flat")
    if args.input_space == "flat":
        # raster flat maps
        # patch size is flexible, since these are images
        args.img_size = (224, 560)
    elif args.input_space == "schaefer400":
        # schaefer400 parcellation. parcellated activity vectors are thought of as
        # "images" of shape (H, W) = (n_rois, 1). patch size is fixed to 1, since roi
        # order is (somewhat) arbitrary. but you still have the temporal patch size.
        args.img_size = (400, 1)
        args.patch_size = 1
    elif args.input_space == "mni_cortex":
        # the mni cortex models have a nuanced structure. the pipeline is:
        # 1. mask the input mni152 2mm (fsl) volume by the schaefer cortex mask
        # 2. pad dims to be divisible by 8
        # 3. patchify with 8 x 8 x 8 patches
        # 4. drop patches not containing enough cortex voxels
        # 5. rearrange "(d p) (h p) (w p) -> (d h w) (p p p)"
        # this results in 466 valid cube patches of dim 8 x 8 x 8 = 512
        # patch size is therefore fixed.
        args.img_size = (466, 512)
        args.patch_size = (1, 512)
    else:
        raise ValueError(f"input space {args.input_space} not implemented")

    if is_master:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_cfg_path = output_dir / "config.yaml"
        if out_cfg_path.exists():
            prev_cfg = OmegaConf.load(out_cfg_path)
            if args != prev_cfg:
                if not args.get("unsafe_resume"):
                    raise ValueError("current config doesn't match previous config")
                suffix = len(list(output_dir.glob("config.yaml.*")))
                OmegaConf.save(prev_cfg, output_dir / f"config.yaml.{suffix}")
        OmegaConf.save(args, out_cfg_path)

        if args.wandb:
            wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=args.name,
                notes=args.notes,
                config=OmegaConf.to_container(args),
            )

    ut.setup_for_distributed(log_path=output_dir / "log.txt")

    print("pretraining fmri mae")
    print(f"start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"cwd: {Path.cwd()}")
    print(ut.get_sha())
    print("config:", OmegaConf.to_yaml(args), sep="\n")

    # data loaders
    train_loader, eval_loaders = create_data_loaders(args)

    # model
    model = MODELS_DICT[args.model](
        img_size=args.img_size,
        in_chans=args.in_chans,
        patch_size=args.patch_size,
        num_frames=args.num_frames,
        t_patch_size=args.t_patch_size,
        **args.model_kwargs,
    )
    model.to(device)
    print("model:", model, sep="\n")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"num params: {num_params / 1e6:.1f}M")

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # todo: compile?

    # optimizer
    total_batch_size = args.batch_size * args.accum_iter * world_size
    print(
        f"total batch size: {total_batch_size} = "
        f"{args.batch_size} bs per gpu x {args.accum_iter} accum x {world_size} gpus"
    )

    if not args.get("lr"):
        args.lr = args.base_lr * total_batch_size / 256
        print(f"lr: {args.lr:.2e} = {args.base_lr:.2e} x {total_batch_size} / 256")
    else:
        print(f"lr: {args.lr:.2e}")

    param_groups = ut.get_param_groups(model)
    ut.update_lr(param_groups, args.lr)
    ut.update_wd(param_groups, args.weight_decay)
    # cast or else it corrupts the checkpoint
    betas = tuple(args.betas) if args.betas is not None else None
    optimizer = torch.optim.AdamW(param_groups, betas=betas)

    epoch_num_batches = len(train_loader)
    steps_per_epoch = epoch_num_batches // args.accum_iter
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    lr_schedule = ut.WarmupThenCosine(
        base_value=args.lr,
        final_value=args.min_lr,
        total_iters=total_steps,
        warmup_iters=warmup_steps,
    )
    print(f"full schedule: epochs = {args.epochs} (steps = {total_steps})")
    print(f"warmup: epochs = {args.warmup_epochs} (steps = {warmup_steps})")

    # loss scaling not needed for bfloat16 (according to timm)
    if args.amp and args.amp_dtype != "bfloat16":
        loss_scaler = torch.GradScaler(device.type)
    else:
        loss_scaler = None

    # load checkpoint/resume training
    ut.load_model(args, model_without_ddp, optimizer, loss_scaler)

    print(f"start training for {args.epochs} epochs")
    start_time = time.monotonic()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and hasattr(train_loader, "sampler"):
            train_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            args,
            model,
            train_loader,
            optimizer,
            loss_scaler,
            lr_schedule,
            epoch,
            device,
        )

        eval_stats = {}
        eval_plots = {}
        for name, loader in eval_loaders.items():
            stats, plots = evaluate(
                args,
                model,
                loader,
                epoch,
                device,
                eval_name=name,
            )
            eval_stats.update(stats)
            eval_plots.update(plots)

        merged_stats = {"epoch": epoch, **train_stats, **eval_stats}
        if is_master:
            with (output_dir / "log.json").open("a") as f:
                print(json.dumps(merged_stats), file=f)

            for plot_name, img in eval_plots.items():
                plot_name = plot_name.replace("/", "__")
                img.save(output_dir / f"{plot_name}__{epoch:05d}.png")

        ut.save_model(args, epoch, model_without_ddp, optimizer, loss_scaler)

    if args.distributed:
        torch.distributed.destroy_process_group()

    total_time = time.monotonic() - start_time
    print(f"done! training time: {datetime.timedelta(seconds=int(total_time))}")


def create_data_loaders(args: DictConfig):
    train_transform = transforms.Transform(
        space=args.input_space,
        num_frames=args.num_frames,
        normalize=args.normalize,
        clip_vmax=args.clip_vmax,
        tr_scale=args.get("tr_scale"),
        crop_scale=args.get("crop_scale"),
        crop_aspect=args.get("crop_aspect"),
        gray_jitter=args.get("gray_jitter"),
        gauss_sigma=args.get("gauss_sigma"),
    )
    val_transform = transforms.Transform(
        space=args.input_space,
        num_frames=args.num_frames,
        normalize=args.normalize,
        clip_vmax=args.clip_vmax,
    )
    print("train transform:", train_transform, sep="\n")
    print("val transform:", val_transform, sep="\n")

    # number of frames needed for train transform random tr scaling
    tr_scale = args.get("tr_scale")
    if tr_scale:
        train_num_frames = round(args.num_frames / tr_scale)
    else:
        train_num_frames = args.num_frames

    # masking generator
    # generate masks during collate, following capi
    # TODO: why do we need to do this again?
    if args.masking:
        # decouple mask patch size from model patch size, pixio style
        mask_patch_size = args.get("mask_patch_size") or args.patch_size
        mask_fn = masking.create_masking(
            args.masking,
            mask_ratio=args.mask_ratio,
            img_size=args.img_size,
            patch_size=mask_patch_size,
            num_frames=args.num_frames,
            t_patch_size=args.t_patch_size,
            **args.masking_kwargs,
        )
        print("mask generator:", mask_fn, sep="\n")
    else:
        print("not using custom masking")
        mask_fn = None

    # mask collate needed even if mask_fn is None to pad the masks to the right shape
    collate_fn = partial(masking.mask_collate, mask_fn=mask_fn)

    data_loaders = {}
    dataset_names = [args.train_dataset] + args.eval_datasets

    for dataset_name in dataset_names:
        dataset_config = args.datasets[dataset_name].copy()
        print(f"loading dataset: {dataset_name}\n\n{OmegaConf.to_yaml(dataset_config)}")

        # we apply train transform to any dataset with train in the name. (this
        # is just to visualize the effect of the train transforms)
        if dataset_name == args.train_dataset or "train" in dataset_name:
            transform = train_transform
        else:
            transform = val_transform

        dataset_type = dataset_config.pop("type")

        if dataset_type == "wds":
            samples_per_epoch = dataset_config.pop("samples_per_epoch")
            dataset = flat_data.make_fmri_wds_dataset(num_frames=train_num_frames, **dataset_config)
            dataset = dataset.map(transform)
            sampler = None
            # the shuffle happens inside the dataset with a buffer.
            shuffle = False
        elif dataset_type == "arrow":
            # note, the arrow datasets are pre-clipped. this means the clips have to be
            # >= num_frames for val and >= train_num_frames for train (including any
            # "eval" subsets of the training set).
            dataset = hfds.load_dataset(
                "arrow",
                data_files=f"{dataset_config.root}/*.arrow",
                split="train",
                download_config=hfds.DownloadConfig(
                    num_proc=min(args.num_workers, 8)
                ),  # parallelize download
            )
            dataset = flat_data.HFDataset(dataset, transform)

            # subset split
            split_range = dataset_config.get("split_range")
            if split_range is not None:
                split_start, split_stop = split_range
                if isinstance(split_stop, float):
                    split_start = int(split_start * len(dataset))
                    split_stop = int(split_stop * len(dataset))
                shuffle_seed = dataset_config.get("shuffle_seed", 42)
                rng = np.random.default_rng(shuffle_seed)
                sample_order = rng.permutation(len(dataset))
                split_indices = sample_order[split_start:split_stop]
                print(f"split indices: {split_indices[:10].tolist()}")
                dataset = Subset(dataset, split_indices)

            if args.distributed:
                sampler = DistributedSampler(dataset, shuffle=dataset_config.shuffle)
            else:
                sampler = None
            samples_per_epoch = len(dataset)
            shuffle = sampler is None and dataset_config.shuffle
        else:
            raise ValueError(f"Unknown dataset type {dataset_type}.")

        loader = WebLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        # setting the epoch length is needed for infinite wds loaders
        num_batches = samples_per_epoch // (ut.get_world_size() * args.batch_size)
        loader = loader.with_epoch(num_batches)
        loader = loader.with_length(num_batches, silent=True)

        data_loaders[dataset_name] = loader

    train_loader = data_loaders.pop(args.train_dataset)
    return train_loader, data_loaders


def train_one_epoch(
    args: DictConfig,
    model: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    loss_scaler: torch.GradScaler | None,
    lr_schedule: Sequence[float],
    epoch: int,
    device: torch.device,
):
    model.train()

    metric_logger = ut.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", ut.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("grad", ut.SmoothedValue())
    header = f"Train: [{epoch}]"
    log_wandb = args.wandb and ut.is_main_process()

    epoch_num_batches = len(data_loader)
    steps_per_epoch = epoch_num_batches // args.accum_iter

    use_cuda = device.type == "cuda"
    if use_cuda and args.presend_cuda:
        data_loader = ut.pre_send_to_cuda_wrapper(data_loader, device)

    print_freq = args.get("print_freq", 100) if not args.debug else 1
    num_batches = epoch_num_batches if not args.debug else 10

    amp_dtype = getattr(torch, args.amp_dtype)

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header, total_steps=num_batches)
    ):
        if use_cuda and not args.presend_cuda:
            batch = ut.send_data(batch, device)

        batch_step = batch_idx + 1
        global_step = epoch * steps_per_epoch + batch_step // args.accum_iter
        lr = lr_schedule[global_step]
        need_update = batch_step % args.accum_iter == 0

        if need_update:
            ut.update_lr(optimizer.param_groups, lr)

        images = batch["bold"]  # note, this key changed "image" -> "bold"

        img_mask = batch["mask"]  # note, this key changed "img_mask" -> "mask"
        targets = batch.get("bold_clean")
        visible_mask = batch.get("visible_mask")

        # visible mask overrides default random masking
        mask_ratio = args.mask_ratio if visible_mask is None else None

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp):
            loss = model(
                images,
                targets=targets,
                img_mask=img_mask,
                visible_mask=visible_mask,
                mask_ratio=mask_ratio,
                pred_mask_ratio=args.pred_mask_ratio,
                with_state=False,
            )

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Loss is {loss_value}, stopping training")

        grad_norm = ut.backward_step(
            loss / args.accum_iter,
            optimizer,
            scaler=loss_scaler,
            need_update=need_update,
            max_norm=args.clip_grad,
        )

        metric_logger.update(loss=loss_value)
        if need_update:
            metric_logger.update(lr=lr)
            grad_norm_value = grad_norm.item()
            metric_logger.update(grad=grad_norm_value)

        if log_wandb:
            log_stats = {"train/loss": loss_value}
            if need_update:
                log_stats.update({"train/lr": lr, "train/grad": grad_norm_value})
            wandb.log(log_stats, step=int(1000 * (epoch + batch_step / epoch_num_batches)))

        if use_cuda:
            torch.cuda.synchronize()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {f"train/{k}": meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.inference_mode()
def evaluate(
    args: DictConfig,
    model: nn.Module,
    data_loader: Iterable,
    epoch: int,
    device: torch.device,
    eval_name: str,
):
    model.eval()

    metric_logger = ut.MetricLogger(delimiter="  ")
    header = f"Eval ({eval_name}): [{epoch}]"
    log_wandb = args.wandb and ut.is_main_process()

    epoch_num_batches = len(data_loader)

    use_cuda = device.type == "cuda"
    if use_cuda and args.presend_cuda:
        data_loader = ut.pre_send_to_cuda_wrapper(data_loader, device)

    print_freq = args.get("print_freq", 100) if not args.debug else 1
    num_batches = epoch_num_batches if not args.debug else 10
    example_step = random.randint(1, num_batches)

    amp_dtype = getattr(torch, args.amp_dtype)

    for batch_idx, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header, total_steps=num_batches)
    ):
        if use_cuda and not args.presend_cuda:
            batch = ut.send_data(batch, device)

        batch_step = batch_idx + 1

        images = batch["bold"]
        img_mask = batch["mask"]
        targets = batch.get("bold_clean")
        visible_mask = batch.get("visible_mask")

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp):
            loss, state = model(
                images,
                targets=targets,
                mask_ratio=args.mask_ratio,
                pred_mask_ratio=args.pred_mask_ratio,
                img_mask=img_mask,
                visible_mask=visible_mask,
            )

        metric_logger.update(loss=loss)

        if batch_step == example_step:
            example_data = {
                "batch": ut.send_data(batch, "cpu"),
                "state": ut.send_data(state, "cpu"),
            }

        if use_cuda:
            torch.cuda.synchronize()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(f"Averaged stats ({eval_name}):", metric_logger)
    stats = {f"eval/{eval_name}/{k}": meter.global_avg for k, meter in metric_logger.meters.items()}

    plot_period = args.get("plot_period", 1)
    if plot_period and (epoch + 1) % plot_period == 0:
        print(f"Making plots ({eval_name}): example={example_step}")
        plots = make_plots(args, **example_data)
        plots = {f"eval/{eval_name}/{k}": img for k, img in plots.items()}
    else:
        plots = {}

    if log_wandb:
        wandb.log(stats, step=1000 * (epoch + 1))
        wandb.log(
            {k: wandb.Image(img, caption=f"example={example_step}") for k, img in plots.items()},
            step=1000 * (epoch + 1),
        )
    return stats, plots


def make_plots(
    args: DictConfig,
    batch: dict[str, Tensor],
    state: dict[str, Tensor],
) -> dict[str, Image.Image]:
    # unmasking transform has a to_flat method that we use for consistent visualization
    # nb this returns a cached transform, doesn't create a new one every time
    unmask = transforms.get_unmask(args.input_space)

    # nb, we omit the pred_mask so we can visualize the full prediction, not just what
    # contributes to the loss. this is mostly for when pred_edge_pad > 0, so we can
    # still see the edge prediction.
    plot_state = {
        "target": batch["bold"],
        "img_mask": batch["mask"],
        "pred": state["pred_images"],
        "visible_mask": state["visible_mask"],
    }
    for key, values in plot_state.items():
        if values is not None:
            # only get first sample from batch, saves transform time
            values = values[:1]
            values = unmask.to_flat(values)
            plot_state[key] = values

    # hack, input mask might still be unexpanded. other masks are fine though.
    plot_state["img_mask"] = plot_state["img_mask"].expand_as(plot_state["target"])

    fig_kwargs = args.get("fig_kwargs", {})
    fig_kwargs = ut.filter_kwargs(vis.plot_mask_pred, fig_kwargs)

    plots = {}
    mask_pred_fig = vis.plot_mask_pred(**plot_state, **fig_kwargs)
    plots["mask_pred"] = vis.fig2pil(mask_pred_fig)
    plt.close(mask_pred_fig)

    return plots


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default=None)
    parser.add_argument("--overrides", type=str, default=None, nargs="+")
    args = parser.parse_args()
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    if args.cfg_path:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.load(args.cfg_path))
    if args.overrides:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(args.overrides))
    main(cfg)
