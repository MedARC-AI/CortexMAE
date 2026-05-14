import math
import urllib.parse
import platformdirs
import fsspec
import shutil
import tempfile
from functools import cache
from pathlib import Path
from typing import Literal, NamedTuple, Protocol

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from torch import Tensor
from einops import rearrange
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, OmegaConf

import cortex_mae.models_mae as models_mae
import cortex_mae.nisc as nisc
import cortex_mae.transforms as transforms
import cortex_mae.masking as masking

CACHE_DIR = platformdirs.user_cache_path("cortexmae")


class EmbeddingOutput(NamedTuple):
    cls_embeds: Tensor | None
    """cls embeddings [B N 1 D]"""

    reg_embeds: Tensor | None
    """register embeddings [B N R D]"""

    patch_embeds: Tensor
    """patch embeddings [B N L D]"""


class ReconstructionOutput(NamedTuple):
    loss: Tensor
    """MAE MSE loss"""

    images: Tensor
    """input images [B C N T H W]"""

    pred_images: Tensor
    """predicted images [B C N T H W]"""

    img_mask: Tensor
    """valid image mask [B C N T H W]"""

    visible_mask: Tensor
    """observed image mask [B C N T H W]"""

    pred_mask: Tensor
    """prediction image mask [B C N T H W]"""


class DenoisingOutput(NamedTuple):
    loss: Tensor
    """MAE MSE loss"""

    images: Tensor
    """input images [B S C N T H W]"""

    pred_images: Tensor
    """predicted images [B S C N T H W]"""

    img_mask: Tensor
    """valid image mask [B S C N T H W]"""

    visible_mask: Tensor
    """observed image mask [B S C N T H W]"""

    pred_mask: Tensor
    """prediction image mask [B S C N T H W]"""

    pred_mean: Tensor
    """denoising prediction mean [B C N T H W]"""

    pred_std: Tensor
    """denoising prediction stdev [B C N T H W]"""


class CortexMAE:
    def __init__(
        self,
        args: DictConfig,
        model: models_mae.MaskedAutoencoderViT,
        reader: "Reader",
        transform: "Transform",
        mask_fn: masking.RandomMasking | None,
    ):
        super().__init__()
        self.args = args
        self.model = model
        self.reader = reader
        self.transform = transform
        self.mask_fn = mask_fn

    @staticmethod
    def from_config(args: DictConfig) -> "CortexMAE":
        reader = get_reader(args.input_space)
        transform = Transform(
            args.input_space,
            norm=args.normalize,
            clip_vmax=args.clip_vmax,
            no_coord_normalize=args.get("no_coord_normalize", False),
        )
        mask_patch_size = args.get("mask_patch_size") or args.patch_size
        mask_fn = masking.create_masking(
            args.masking or "random",
            mask_ratio=args.mask_ratio,
            img_size=args.img_size,
            patch_size=mask_patch_size,
            num_frames=args.num_frames,
            t_patch_size=args.t_patch_size,
        )
        model_fn = getattr(models_mae, args.model)
        model: torch.nn.Module = model_fn(
            img_size=args.img_size,
            in_chans=args.in_chans,
            patch_size=args.patch_size,
            num_frames=args.num_frames,
            t_patch_size=args.t_patch_size,
            **args.model_kwargs,
        )
        model.eval()
        model = CortexMAE(
            args=args,
            model=model,
            reader=reader,
            transform=transform,
            mask_fn=mask_fn,
        )
        return model

    @staticmethod
    def from_checkpoint(ckpt_path: str) -> "CortexMAE":
        ckpt_path = resolve_checkpoint(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        args = OmegaConf.create(ckpt["args"])
        model = CortexMAE.from_config(args)
        model.model.load_state_dict(ckpt["model"])
        return model

    @staticmethod
    def from_pretrained(model_name: str) -> "CortexMAE":
        ckpt_path = CORTEX_MAE_MODEL_REGISTRY[model_name]
        ckpt_path = f"{HF_PREFIX}/{ckpt_path}"
        return CortexMAE.from_checkpoint(ckpt_path)

    def set_device(self, device: torch.device | None = None) -> "CortexMAE":
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        return self

    def get_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def load_input(
        self,
        input: str | Path | dict[str, Tensor],
        *,
        tr: float | None = None,
    ) -> dict[str, Tensor]:
        if isinstance(input, (str, Path)):
            with tempfile.TemporaryDirectory(prefix="cortexmae-") as tmpdir:
                path = resolve_file(input, cache_dir=tmpdir)
                sample = read_sample(self.reader, path, tr=tr)
        elif isinstance(input, dict):
            sample = input
        else:
            raise TypeError(f"Invalid input {type(input)}")
        return sample

    @torch.inference_mode()
    def run_embedding(
        self,
        input: str | Path | dict[str, Tensor],
        *,
        tr: float | None = None,
    ) -> "EmbeddingOutput":
        sample = self.load_input(input, tr=tr)
        sample = self.transform(sample)
        bold, mask = unpack_batch(sample, device=self.get_device())
        embeds = forward_embedding(self.model.encoder, bold, mask)
        return embeds

    @torch.inference_mode()
    def run_masked_recon(
        self,
        input: str | Path | dict[str, Tensor],
        *,
        tr: float | None = None,
        mask_ratio: float | None = None,
        batch_size: int | None = None,
    ) -> "ReconstructionOutput":
        sample = self.load_input(input, tr=tr)
        sample = self.transform(sample)
        bold, mask = unpack_batch(sample, device=self.get_device())
        state = forward_masked_recon(
            self.model,
            bold,
            mask,
            mask_fn=self.mask_fn,
            mask_ratio=mask_ratio,
            batch_size=batch_size,
        )
        return ReconstructionOutput(**state)

    @torch.inference_mode()
    def run_denoise(
        self,
        input: str | Path | dict[str, Tensor],
        *,
        tr: float | None = None,
        num_samples: int = 100,
        mask_ratio: float | None = None,
        pred_edge_pad: int | None = None,
        batch_size: int | None = None,
    ):
        assert num_samples > 1, f"denoising requires num_samples > 1, got {num_samples}"
        sample = self.load_input(input, tr=tr)
        sample = self.transform(sample)
        bold, mask = unpack_batch(sample, device=self.get_device())
        state = forward_masked_recon(
            self.model,
            bold,
            mask,
            mask_fn=self.mask_fn,
            mask_ratio=mask_ratio,
            num_samples=num_samples,
            batch_size=batch_size,
            pred_edge_pad=pred_edge_pad,
        )
        return DenoisingOutput(**state)


@torch.inference_mode()
def forward_masked_recon(
    model: models_mae.MaskedAutoencoderViT,
    bold: Tensor,
    mask: Tensor,
    *,
    mask_fn: masking.RandomMasking,
    mask_ratio: float | None = None,
    pred_edge_pad: int | None = None,
    num_samples: int | None = None,
    batch_size: int | None = None,
) -> dict[str, Tensor]:
    num_frames = model.encoder.patchify.img_size[0]

    # repeat for multi-sample reconstruction
    if num_samples:
        bold = torch.repeat_interleave(bold, num_samples, dim=0)
        mask = torch.repeat_interleave(mask, num_samples, dim=0)

    # pad/truncate and unfold into non-overlapping sliding windows
    bold, mask, num_clips = pad_unfold(bold, mask, num_frames=num_frames)

    # B = (batch_size * num_samples * num_clips)
    B, C, T, H, W = bold.shape

    # random masks
    visible_mask = torch.zeros_like(mask)
    for ii in range(B):
        visible_mask[ii] = mask_fn(mask[ii, 0, 0], mask_ratio=mask_ratio)

    if batch_size is None:
        batch_size = B
    num_batches = math.ceil(B / batch_size)
    keys = ["images", "pred_images", "img_mask", "visible_mask", "pred_mask"]
    state = {k: [] for k in ["loss"] + keys}

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        stop = start + batch_size
        loss, batch_state = model.forward(
            bold[start:stop],
            img_mask=mask[start:stop],
            visible_mask=visible_mask[start:stop],
            mask_ratio=None,
            pred_edge_pad=pred_edge_pad,
        )
        state["loss"].append(loss)
        for k in keys:
            state[k].append(batch_state[k])

    state["loss"] = torch.stack(state["loss"]).mean()
    for k in keys:
        v = torch.cat(state[k])
        if num_samples:
            v = rearrange(v, "(b s n) c t h w -> b s c n t h w", s=num_samples, n=num_clips)
        else:
            v = rearrange(v, "(b n) c t h w -> b c n t h w", n=num_clips)
        state[k] = v

    if num_samples and num_samples > 1:
        pred_images: Tensor = state["pred_images"]
        pred_mask: Tensor = state["pred_mask"]
        pred_images = pred_images * pred_mask
        pred_count = pred_mask.sum(dim=1, keepdim=True)
        pred_mean = pred_images.sum(dim=1, keepdim=True) / pred_count.clip(min=1)
        pred_std = (((pred_images - pred_mean) ** 2).sum(dim=1) / pred_count.clip(min=1)).sqrt()
        state["pred_mean"] = pred_mean
        state["pred_std"] = pred_std

    return state


def forward_embedding(
    encoder: models_mae.MaskedEncoder,
    bold: Tensor,
    mask: Tensor,
) -> EmbeddingOutput:
    num_frames = encoder.patchify.img_size[0]

    # pad/truncate and unfold into non-overlapping sliding windows
    bold, mask, num_clips = pad_unfold(bold, mask, num_frames=num_frames)

    cls_embeds, reg_embeds, patch_embeds = encoder.forward_embedding(bold, mask)

    # unflatten batch and clip dimensions
    if cls_embeds is not None:
        cls_embeds = rearrange(cls_embeds, "(b n) l d -> b n l d", n=num_clips)
    if reg_embeds is not None:
        reg_embeds = rearrange(reg_embeds, "(b n) l d -> b n l d", n=num_clips)
    patch_embeds = rearrange(patch_embeds, "(b n) l d -> b n l d", n=num_clips)

    return EmbeddingOutput(cls_embeds, reg_embeds, patch_embeds)


def unpack_batch(
    batch: dict[str, Tensor], device: torch.device | None = None
) -> tuple[Tensor, Tensor]:
    bold = batch["bold"]
    mask = batch["mask"]
    if device is not None:
        bold = bold.to(device)
        mask = mask.to(device)
    # normalize shapes to [B, C, T, H, W]
    if bold.ndim == 4:
        bold = bold[None]
    if mask.ndim == 2:
        mask = mask[None, None, None, :, :]
    elif mask.ndim == 3:
        mask = mask[:, None, None, :, :]
    mask = mask.expand_as(bold)
    return bold, mask


def pad_unfold(
    bold: torch.Tensor, mask: torch.Tensor, num_frames: int = 16
) -> tuple[torch.Tensor, torch.Tensor, int]:
    B, C, T, H, W = bold.shape
    assert bold.shape == mask.shape

    # pad inputs that are too short
    # padding the mask excludes the patches from the forward pass
    if T < num_frames:
        pad = num_frames - T
        bold = F.pad(bold, (0, 0, 0, 0, 0, pad))
        mask = F.pad(mask, (0, 0, 0, 0, 0, pad))
        T = num_frames

    # truncate to divisible by num frames
    # nb, we have to truncate, we can't pad bc all samples in batch need the same number
    # of valid patches.
    num_clips = T // num_frames
    bold = bold[:, :, : num_clips * num_frames]
    mask = mask[:, :, : num_clips * num_frames]

    # rearrange into a batch of clips and apply model as sliding window.
    if num_clips > 1:
        bold = rearrange(bold, "b c (n f) h w -> (b n) c f h w", n=num_clips)
        mask = rearrange(mask, "b c (n f) h w -> (b n) c f h w", n=num_clips)
    return bold, mask, num_clips


def resolve_checkpoint(path: str) -> str:
    path = str(path)
    if path.startswith("hf://"):
        namespace, repo, filename = path[len("hf://") :].split("/", 2)
        return hf_hub_download(repo_id=f"{namespace}/{repo}", filename=filename)
    return path


def resolve_file(path: str | Path, cache_dir: str | Path | None = None, **kwargs) -> Path:
    path = str(path)
    parsed = urllib.parse.urlparse(path)

    if not parsed.scheme:
        return Path(path)

    cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
    local_path = cache_dir / parsed.netloc / parsed.path.lstrip("/")
    if local_path.exists():
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)

    with fsspec.open(path, "rb", **kwargs) as fsrc:
        with local_path.open("wb") as fdst:
            shutil.copyfileobj(fsrc, fdst)

    return local_path


class Reader(Protocol):
    def __call__(self, path: str) -> np.ndarray: ...

    def to_image(
        self, series: np.ndarray, tr: float
    ) -> nib.Nifti1Image | nib.cifti2.Cifti2Image: ...


class FlatReader:
    def __init__(self):
        self.resampler = nisc.flat_resampler_fslr64k_224_560()

    def __call__(self, path: str) -> np.ndarray:
        if str(path).endswith(".gii"):
            series = nisc.read_gifti_surf_data(path)
        else:
            series = nisc.read_cifti_surf_data(path)
        series = self.resampler.transform(series, interpolation="linear")
        series = series[:, self.resampler.mask_]
        return series

    def to_image(self, series: np.ndarray, tr: float) -> nib.cifti2.Cifti2Image:
        T, D = series.shape
        series_ = np.zeros((T, *self.resampler.mask_.shape), dtype=series.dtype)
        series_[:, self.resampler.mask_] = series
        series = series_
        series = self.resampler.inverse(series)
        img = nisc.make_cifti_fslr64k_img(series, tr)
        return img


class SchaeferReader:
    def __init__(self, num_rois: int = 400):
        path = nisc.fetch_schaefer(num_rois, space="fslr64k")
        self.parc = nisc.read_cifti_surf_data(path).squeeze(0)
        self.parcavg = nisc.ParcelAverage(self.parc)

    def __call__(self, path: str) -> np.ndarray:
        if str(path).endswith(".gii"):
            series = nisc.read_gifti_surf_data(path)
        else:
            series = nisc.read_cifti_surf_data(path)
        series = self.parcavg(series)
        return series

    def to_image(self, series: np.ndarray, tr: float) -> nib.cifti2.Cifti2Image:
        # parcellation codes 0 as background and 1-indexed rois
        parc_mask = self.parc > 0
        parc_ids = self.parc - 1
        series = series[..., parc_ids] * parc_mask
        img = nisc.make_cifti_fslr64k_img(series, tr)
        return img


class MNICortexReader:
    def __init__(self):
        path = nisc.fetch_schaefer(400, space="mni")
        self.mask = nisc.read_mni152_2mm_data(path, interpolation="nearest") > 0

    def __call__(self, path: str) -> np.ndarray:
        series = nisc.read_mni152_2mm_data(path, interpolation="linear")
        series = series[:, self.mask]
        return series

    def to_image(self, series: np.ndarray, tr: float) -> nib.nifti1.Nifti1Image:
        T, D = series.shape
        series_ = np.zeros((T, *self.mask.shape), dtype=series.dtype)
        series_[:, self.mask] = series
        series = series_
        img = nisc.make_mni152_img(series, tr)
        return img


class SchaeferTianReader:
    def __init__(self, num_rois: int = 400, scale: int = 3):
        path = nisc.fetch_schaefer_tian(num_rois, scale, space="fslr91k")
        self.parc = nisc.read_cifti_data(path).squeeze(0)
        self.parcavg = nisc.ParcelAverage(self.parc)

    def __call__(self, path: str) -> np.ndarray:
        series = nisc.read_cifti_surf_data(path)
        series = self.parcavg(series)
        return series

    def to_image(self, series: np.ndarray, tr: float) -> nib.cifti2.Cifti2Image:
        # parcellation codes 0 as background and 1-indexed rois
        parc_mask = self.parc > 0
        parc_ids = self.parc - 1
        series = series[..., parc_ids] * parc_mask
        img = nisc.make_cifti_fslr91k_img(series, tr)
        return img


class A424Reader:
    def __init__(self):
        path = nisc.fetch_a424(cifti=True)
        self.parc = nisc.read_cifti_data(path).squeeze(0)
        self.parcavg = nisc.ParcelAverage(self.parc)

    def __call__(self, path: str) -> np.ndarray:
        series = nisc.read_cifti_surf_data(path)
        series = self.parcavg(series)
        return series

    def to_image(self, series: np.ndarray, tr: float) -> nib.cifti2.Cifti2Image:
        # parcellation codes 0 as background and 1-indexed rois
        parc_mask = self.parc > 0
        parc_ids = self.parc - 1
        series = series[..., parc_ids] * parc_mask
        img = nisc.make_cifti_fslr91k_img(series, tr)
        return img


def read_sample(
    reader: Reader, path: str | Path, *, tr: float | None = None
) -> dict[str, np.ndarray]:
    path = str(path)
    series = reader(path)
    if tr is None:
        tr = nisc.get_tr(path)
    series, mean, std = nisc.scale(series)
    sample = {
        "bold": series.astype(np.float16),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "tr": float(tr),
    }
    return sample


@cache
def get_reader(space: str = "flat") -> Reader:
    reader_cls = {
        "flat": FlatReader,
        "schaefer400": SchaeferReader,
        "mni_cortex": MNICortexReader,
        "schaefer400_tians3": SchaeferTianReader,
        "a424": A424Reader,
    }[space]
    reader = reader_cls()
    return reader


class Transform:
    def __init__(
        self,
        space: Literal["schaefer400", "flat", "mni_cortex"] = "flat",
        norm: Literal["frame", "global"] | None = "frame",
        clip_vmax: float | None = 3.0,
        no_coord_normalize: bool = False,
    ):
        super().__init__()
        self.norm = norm
        self.clip_vmax = clip_vmax
        self.target_tr = 1.0
        self.no_coord_normalize = no_coord_normalize
        self.unmask = transforms.get_unmask(space)

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        bold = sample["bold"]
        mean = sample["mean"]
        std = sample["std"]
        tr = float(sample["tr"])

        if self.no_coord_normalize:
            bold = bold * std + mean

        bold = torch.as_tensor(bold, dtype=torch.float32)

        # temporal resample
        # nb, pretraining data used pchip interpolation, but that's very slow.
        if abs(tr - self.target_tr) > 0.1:
            bold = resample_to_tr(bold, tr=tr, target_tr=self.target_tr, mode="linear")

        # sample-wise normalization
        if self.norm:
            dim = {"frame": 1, "global": None}[self.norm]
            bold = normalize(bold, dim=dim)

        # clipping
        if self.clip_vmax and self.clip_vmax > 0:
            bold = torch.clamp(bold, min=-self.clip_vmax, max=self.clip_vmax)

        # unmask masked input
        sample["bold"] = bold
        sample = self.unmask(sample)
        return sample


def normalize(x: torch.Tensor, dim: int | None = None, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True)
    x = (x - mean) / (std + eps)
    return x


def resample_to_tr(
    x: torch.Tensor, tr: float, target_tr: float, mode: str = "linear"
) -> torch.Tensor:
    T, D = x.shape
    x = x.t().unsqueeze(0)  # [1, D, T]
    x = F.interpolate(x, size=round(tr * T / target_tr), mode=mode)
    x = x.squeeze(0).t()
    return x


HF_PREFIX = "hf://medarc/CortexMAE"


CORTEX_MAE_MODEL_REGISTRY = {
    # input space
    "cortex_mae_flat": "input_space_v3/flat_lr1e-3_1/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_r2": "input_space_v3/flat_lr1e-3_2/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_r3": "input_space_v3/flat_lr1e-3_3/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_r4": "input_space_v3/flat_lr1e-3_4/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_r5": "input_space_v3/flat_lr1e-3_5/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_r6": "input_space_v3/flat_lr1e-3_6/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_r7": "input_space_v3/flat_lr1e-3_7/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_r8": "input_space_v3/flat_lr1e-3_8/pretrain/checkpoint-last.pth",
    "cortex_mae_volume": "input_space_v3/mni_cortex_lr1e-3_1/pretrain/checkpoint-last.pth",
    "cortex_mae_volume_r2": "input_space_v3/mni_cortex_lr1e-3_2/pretrain/checkpoint-last.pth",
    "cortex_mae_volume_r3": "input_space_v3/mni_cortex_lr1e-3_3/pretrain/checkpoint-last.pth",
    "cortex_mae_volume_r4": "input_space_v3/mni_cortex_lr1e-3_4/pretrain/checkpoint-last.pth",
    "cortex_mae_volume_r5": "input_space_v3/mni_cortex_lr1e-3_5/pretrain/checkpoint-last.pth",
    "cortex_mae_volume_r6": "input_space_v3/mni_cortex_lr1e-3_6/pretrain/checkpoint-last.pth",
    "cortex_mae_volume_r7": "input_space_v3/mni_cortex_lr1e-3_7/pretrain/checkpoint-last.pth",
    "cortex_mae_volume_r8": "input_space_v3/mni_cortex_lr1e-3_8/pretrain/checkpoint-last.pth",
    "cortex_mae_parcel": "input_space_v3/schaefer400_lr3e-4_1/pretrain/checkpoint-last.pth",
    "cortex_mae_parcel_r2": "input_space_v3/schaefer400_lr3e-4_2/pretrain/checkpoint-last.pth",
    "cortex_mae_parcel_r3": "input_space_v3/schaefer400_lr3e-4_3/pretrain/checkpoint-last.pth",
    "cortex_mae_parcel_r4": "input_space_v3/schaefer400_lr3e-4_4/pretrain/checkpoint-last.pth",
    "cortex_mae_parcel_r5": "input_space_v3/schaefer400_lr3e-4_5/pretrain/checkpoint-last.pth",
    "cortex_mae_parcel_r6": "input_space_v3/schaefer400_lr3e-4_6/pretrain/checkpoint-last.pth",
    "cortex_mae_parcel_r7": "input_space_v3/schaefer400_lr3e-4_7/pretrain/checkpoint-last.pth",
    "cortex_mae_parcel_r8": "input_space_v3/schaefer400_lr3e-4_8/pretrain/checkpoint-last.pth",
    # subcortical
    "cortex_mae_a424": "subcortical/a424_lr3e-4_1/pretrain/checkpoint-last.pth",
    "cortex_mae_a424_r2": "subcortical/a424_lr3e-4_2/pretrain/checkpoint-last.pth",
    "cortex_mae_a424_r3": "subcortical/a424_lr3e-4_3/pretrain/checkpoint-last.pth",
    "cortex_mae_a424_r4": "subcortical/a424_lr3e-4_4/pretrain/checkpoint-last.pth",
    "cortex_mae_s400ts3": "subcortical/schaefer400_tians3_lr3e-4_1/pretrain/checkpoint-last.pth",
    "cortex_mae_s400ts3_r2": "subcortical/schaefer400_tians3_lr3e-4_2/pretrain/checkpoint-last.pth",
    "cortex_mae_s400ts3_r3": "subcortical/schaefer400_tians3_lr3e-4_3/pretrain/checkpoint-last.pth",
    "cortex_mae_s400ts3_r4": "subcortical/schaefer400_tians3_lr3e-4_4/pretrain/checkpoint-last.pth",
    # data scaling
    "cortex_mae_flat_n100": "data_scaling/n100_1/pretrain/checkpoint-best.pth",
    "cortex_mae_flat_n100_r2": "data_scaling/n100_2/pretrain/checkpoint-best.pth",
    "cortex_mae_flat_n200": "data_scaling/n200_1/pretrain/checkpoint-best.pth",
    "cortex_mae_flat_n200_r2": "data_scaling/n200_2/pretrain/checkpoint-best.pth",
    "cortex_mae_flat_n400": "data_scaling/n400_1/pretrain/checkpoint-best.pth",
    "cortex_mae_flat_n400_r2": "data_scaling/n400_2/pretrain/checkpoint-best.pth",
    "cortex_mae_flat_n800": "data_scaling/n800_1/pretrain/checkpoint-best.pth",
    "cortex_mae_flat_n800_r2": "data_scaling/n800_2/pretrain/checkpoint-best.pth",
    "cortex_mae_flat_n1600": "data_scaling/n1600_1/pretrain/checkpoint-best.pth",
    "cortex_mae_flat_n1600_r2": "data_scaling/n1600_2/pretrain/checkpoint-best.pth",
    # model scaling (depth)
    "cortex_mae_flat_d3": "model_scaling/d3/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_d3_r2": "model_scaling/d3_2/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_d6": "model_scaling/d6/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_d6_r2": "model_scaling/d6_2/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_d9": "model_scaling/d9/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_d9_r2": "model_scaling/d9_2/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_d12": "input_space_v3/flat_lr1e-3_1/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_d12_r2": "input_space_v3/flat_lr1e-3_2/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_d15": "model_scaling/d15/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_d15_r2": "model_scaling/d15_2/pretrain/checkpoint-last.pth",
    # temporal patch size
    "cortex_mae_flat_pt1": "t_patch_size/pt-1/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_pt2": "t_patch_size/pt-2/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_pt4": "t_patch_size/pt-4/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_pt8": "t_patch_size/pt-8/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_pt16": "t_patch_size/pt-16/pretrain/checkpoint-last.pth",
    # spatial patch size
    "cortex_mae_flat_p8": "patch_size/patch8/pretrain/checkpoint-last.pth",
    # denoising model
    "cortex_mae_flat_denoise": "decoders/attn_reg1_pep4/pretrain/checkpoint-last.pth",
    # cross-register decoding
    "cortex_mae_flat_crossreg1": "decoders/crossreg_reg1/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_crossreg4": "decoders/crossreg_reg4/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_crossreg16": "decoders/crossreg_reg16/pretrain/checkpoint-last.pth",
    # training data
    "cortex_mae_flat_rest": "rest_v_task/rest_ep50_1/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_rest_r2": "rest_v_task/rest_ep50_2/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_task": "rest_v_task/task_ep50_1/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_task_r2": "rest_v_task/task_ep50_2/pretrain/checkpoint-last.pth",
}


def list_models() -> list[str]:
    return list(CORTEX_MAE_MODEL_REGISTRY)
