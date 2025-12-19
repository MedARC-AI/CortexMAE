import json
from typing import Iterable

import torch
from omegaconf import OmegaConf, DictConfig
from dotenv import load_dotenv
import torch.nn as nn
import webdataset as wds

import data.flat_data as flat_data
import flat_mae.models_mae as models_mae
import flat_mae.utils as ut

MODELS_DICT = models_mae.__dict__

load_dotenv("../../.env")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    args = OmegaConf.load("../../src/flat_mae/config/default_pretrain.yaml")

    model = MODELS_DICT[args.model](
        img_size=args.img_size,
        in_chans=args.in_chans,
        patch_size=args.patch_size,
        num_frames=args.num_frames,
        t_patch_size=args.t_patch_size,
        **args.model_kwargs,
    )
    model = model.to(device)

    ckpt_path = "checkpoints/pretrain_mix/ukbb_n1800/pretrain/checkpoint-last.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    url = "pipe:aws s3 cp s3://sophont/fmri-fm/datasets/ukbb-flat/ukbb-flat_{01800..07903}.tar -"
    dataset = wds.WebDataset(
        url,
        handler=flat_data.warn_and_continue,
        shardshuffle=False,
    )

    dataset = dataset.decode().map(
        flat_data.extract_flat_sample,
        handler=flat_data.warn_and_continue,
    )

    transform = flat_data.make_flat_transform(
        img_size=args.img_size,
        clip_vmax=args.clip_vmax,
        normalize=args.normalize,
    )
    dataset = dataset.map(transform)
    dataset = dataset.map(to_clips)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=16,
    )

    loss_grad_log_path = "results/ukbb_loss_grad.json"

    for ii, record in enumerate(evaluate(args, model, loader, device)):
        with open(loss_grad_log_path, "a") as f:
            print(json.dumps(record), file=f)


def to_clips(sample, n_frames: int = 16):
    img = sample["image"]
    mask = sample["img_mask"]
    C, T, H, W = img.shape
    n_clips = T // n_frames
    img = img[:, : (n_clips * n_frames)].reshape(n_clips, 1, n_frames, H, W)
    mask = mask.expand_as(img)
    return {**sample, "image": img, "img_mask": mask}


def evaluate(
    args: DictConfig,
    model: nn.Module,
    data_loader: Iterable,
    device: torch.device,
):
    model.train()

    metric_logger = ut.MetricLogger(delimiter="  ")
    metric_logger.add_meter("grad", ut.SmoothedValue())

    use_cuda = device.type == "cuda"

    print_freq = 100
    num_batches = 200_000
    header = "Eval"

    for batch_idx, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header, total_steps=num_batches)
    ):
        if use_cuda:
            batch = ut.send_data(batch, device)

        images = batch["image"]
        img_mask = batch.get("img_mask")
        visible_mask = batch.get("visible_mask")

        key = batch["__key__"]
        assert isinstance(key, str)

        # visible mask overrides default random masking
        mask_ratio = args.mask_ratio if visible_mask is None else None

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
            loss = model(
                images,
                img_mask=img_mask,
                visible_mask=visible_mask,
                mask_ratio=mask_ratio,
                pred_mask_ratio=args.pred_mask_ratio,
                with_state=False,
            )
        loss.backward()
        grad = nn.utils.get_total_norm(p.grad for p in model.parameters())
        model.zero_grad()

        loss_value = loss.item()
        grad_value = grad.item()

        metric_logger.update(loss=loss_value)
        metric_logger.update(grad=grad_value)

        record = {
            "key": key,
            "loss": loss_value,
            "grad": grad_value,
        }

        yield record


if __name__ == "__main__":
    main()
