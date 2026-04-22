from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

from fmri_fm_eval.models.base import Embeddings
from fmri_fm_eval.models.registry import register_model

import flat_mae.models_mae as models_mae
import flat_mae.transforms as flat_transforms


class MaskedEncoderWrapper(nn.Module):
    __space__: str = "flat"

    def __init__(self, model: models_mae.MaskedEncoder):
        super().__init__()
        T, H, W = model.patchify.img_size
        self.num_frames = T
        self.model = model

    def forward(self, batch: dict[str, Tensor]) -> Embeddings:
        x = batch["bold"]
        mask = batch["mask"]

        B, C, T, H, W = x.shape

        # pad inputs that are too short
        # padding the mask excludes the patches from the forward pass
        if T < self.num_frames:
            pad = self.num_frames - T
            x = F.pad(x, (0, 0, 0, 0, 0, pad))
            mask = F.pad(mask, (0, 0, 0, 0, 0, pad))
            T = self.num_frames

        # truncate to divisible by num frames
        num_clips = T // self.num_frames
        T = num_clips * self.num_frames
        x = x[:, :, :T]
        mask = mask[:, :, :T]

        # rearrange into a batch of clips and apply model as sliding window.
        if num_clips > 1:
            x = rearrange(x, "b c (n f) h w -> (b n) c f h w", n=num_clips)
            mask = rearrange(mask, "b c (n f) h w -> (b n) c f h w", n=num_clips)

        cls_embeds, reg_embeds, patch_embeds = self.model.forward_embedding(x, mask)

        # rearrange clips back into single seq of embeddings.
        if num_clips > 1:
            if cls_embeds is not None:
                cls_embeds = rearrange(cls_embeds, "(b n) l d -> b (n l) d", n=num_clips)
                cls_embeds = cls_embeds.mean(1, keepdim=True)
            if reg_embeds is not None:
                reg_embeds = rearrange(reg_embeds, "(b n) l d -> b (n l) d", n=num_clips)
            if patch_embeds is not None:
                # nb, this is a lot of tokens. decide if this is really what we want.
                # we could also average pool over some of the grid dims (n, t, h, w).
                patch_embeds = rearrange(patch_embeds, "(b n) l d -> b (n l) d", n=num_clips)

        return cls_embeds, reg_embeds, patch_embeds


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
        self.unmask = flat_transforms.get_unmask(space)

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        bold = sample["bold"]
        mean = sample["mean"]
        std = sample["std"]
        tr = float(sample["tr"])

        if self.no_coord_normalize:
            bold = bold * std + mean

        # temporal resample
        # nb, pretraining data used pchip interpolation, but that's very slow.
        # TODO: we are allowing some tolerance to the tr, but we didn't pretrain with
        # any tr variation. probably should do that, seems like a decent augmentation.
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

        # expand mask to sampe shape as input for correct collation
        sample["mask"] = sample["mask"].expand_as(sample["bold"])
        return sample

    @staticmethod
    def from_checkpoint(ckpt_path: str, no_coord_normalize: bool | None = None) -> "Transform":
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        args = ckpt["args"]
        if no_coord_normalize is None:
            no_coord_normalize = args.get("no_coord_normalize", False)
        transform = Transform(
            space=args["input_space"],
            norm=args["normalize"],
            clip_vmax=args["clip_vmax"],
            no_coord_normalize=no_coord_normalize,
        )
        return transform


def normalize(x: torch.Tensor, dim: int | None = None, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True)
    x = (x - mean) / (std + eps)
    return x


def resample_to_tr(x: Tensor, tr: float, target_tr: float, mode: str = "linear") -> Tensor:
    T, D = x.shape
    x = x.t().unsqueeze(0)  # [1, D, T]
    x = F.interpolate(x, size=round(tr * T / target_tr), mode=mode)
    x = x.squeeze(0).t()
    return x


# TODO: (maybe)
#   - add random flat mae (call init_weights)
#   - patch embed only (stripping off vit blocks)
#   - extract features from different layer using feature extracton
#     https://github.com/MedARC-AI/algonauts2025/blob/main/src/feature_extractor.py


@register_model
def flat_mae_base_patch16_16(**kwargs) -> tuple[Transform, MaskedEncoderWrapper]:
    transform = Transform()
    model = models_mae.MaskedAutoencoderViT.from_pretrained("medarc/fm_mae_vit_base_patch16-16.hcp")
    model = MaskedEncoderWrapper(model.encoder)
    return transform, model


@register_model
def flat_mae_base_patch16_2(**kwargs) -> tuple[Transform, MaskedEncoderWrapper]:
    transform = Transform()
    model = models_mae.MaskedAutoencoderViT.from_pretrained("medarc/fm_mae_vit_base_patch16-2.hcp")
    model = MaskedEncoderWrapper(model.encoder)
    return transform, model


@register_model
def flat_mae(
    *,
    ckpt_path: str,
    no_coord_normalize: bool | None = None,
    scratch_init: bool = False,
    keep_blocks: int | None = None,
    **kwargs,
) -> tuple[Transform, MaskedEncoderWrapper]:
    transform = Transform.from_checkpoint(ckpt_path, no_coord_normalize=no_coord_normalize)
    model = models_mae.MaskedAutoencoderViT.from_checkpoint(ckpt_path, **kwargs)
    # re-init weights to train from scratch
    if scratch_init:
        model.init_weights()
    # remove some vit blocks (nb keep_blocks=0 is patch embed only)
    if keep_blocks is not None:
        model.encoder.blocks = model.encoder.blocks[:keep_blocks]
    model = MaskedEncoderWrapper(model.encoder)
    return transform, model


@register_model
def schaefer400_mae(
    *, ckpt_path: str, no_coord_normalize: bool | None = None, **kwargs
) -> tuple[Transform, MaskedEncoderWrapper]:
    transform = Transform.from_checkpoint(ckpt_path, no_coord_normalize=no_coord_normalize)
    model = models_mae.MaskedAutoencoderViT.from_checkpoint(ckpt_path, **kwargs)
    model = MaskedEncoderWrapper(model.encoder)
    model.__space__ = "schaefer400"
    return transform, model


@register_model
def mni_cortex_mae(
    *, ckpt_path: str, no_coord_normalize: bool | None = None, **kwargs
) -> tuple[Transform, MaskedEncoderWrapper]:
    transform = Transform.from_checkpoint(ckpt_path, no_coord_normalize=no_coord_normalize)
    model = models_mae.MaskedAutoencoderViT.from_checkpoint(ckpt_path, **kwargs)
    model = MaskedEncoderWrapper(model.encoder)
    model.__space__ = "mni_cortex"
    return transform, model


@register_model
def schaefer400_tians3_mae(
    *, ckpt_path: str, no_coord_normalize: bool | None = None, **kwargs
) -> tuple[Transform, MaskedEncoderWrapper]:
    transform = Transform.from_checkpoint(ckpt_path, no_coord_normalize=no_coord_normalize)
    model = models_mae.MaskedAutoencoderViT.from_checkpoint(ckpt_path, **kwargs)
    model = MaskedEncoderWrapper(model.encoder)
    model.__space__ = "schaefer400_tians3"
    return transform, model


@register_model
def a424_mae(
    *, ckpt_path: str, no_coord_normalize: bool | None = None, **kwargs
) -> tuple[Transform, MaskedEncoderWrapper]:
    transform = Transform.from_checkpoint(ckpt_path, no_coord_normalize=no_coord_normalize)
    model = models_mae.MaskedAutoencoderViT.from_checkpoint(ckpt_path, **kwargs)
    model = MaskedEncoderWrapper(model.encoder)
    model.__space__ = "a424"
    return transform, model
