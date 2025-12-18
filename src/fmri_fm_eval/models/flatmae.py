from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

import fmri_fm_eval.nisc as nisc
from fmri_fm_eval.models.base import Embeddings
from fmri_fm_eval.models.registry import register_model

import flat_mae.models_mae as models_mae

# static flat map mask
_resampler = nisc.flat_resampler_fslr64k_224_560()


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

        # pad time dimension to be divisible by num frames
        # padding the mask excludes the patches from the forward pass
        if T % self.num_frames != 0:
            pad = self.num_frames - T % self.num_frames
            x = F.pad(x, (0, 0, 0, 0, 0, pad))
            mask = F.pad(mask, (0, 0, 0, 0, 0, pad))
            T = T + pad

        # rearrange into a batch of clips and apply model as sliding window.
        num_clips = T // self.num_frames
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


class FlatTransform(nn.Module):
    mask: Tensor

    def __init__(
        self,
        norm: Literal["frame", "global"] | None = "frame",
        clip_vmax: float | None = 3.0,
    ):
        super().__init__()
        self.norm = norm
        self.clip_vmax = clip_vmax
        self.register_buffer("mask", torch.tensor(_resampler.mask_))

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        bold = sample["bold"]

        # temporal resample
        # nb, pretraining data used pchip interpolation, but that's very slow.
        bold = nisc.resample_timeseries(bold.numpy(), tr=sample["tr"], new_tr=1.0, kind="linear")
        bold = torch.as_tensor(bold, dtype=torch.float32)

        # sample-wise normalization
        if self.norm:
            dim = {"frame": 1, "global": None}[self.norm]
            bold = normalize(bold, dim=dim)

        # clipping
        if self.clip_vmax and self.clip_vmax > 0:
            bold = torch.clamp(bold, min=-self.clip_vmax, max=self.clip_vmax)

        # unmask masked input
        T, D = bold.shape
        H, W = self.mask.shape
        bold_ = torch.zeros((T, H, W), dtype=bold.dtype)
        bold_[:, self.mask] = bold

        # add channel dim
        bold_ = bold_.unsqueeze(0)  # [C, T, H, W]
        # expand mask to sampe shape as input for correct collation
        mask = self.mask.expand_as(bold_)

        sample = {**sample, "bold": bold_, "mask": mask}
        return sample


def normalize(x: torch.Tensor, dim: int | None = None, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True)
    x = (x - mean) / (std + eps)
    return x


@register_model
def flat_mae_base_patch16_16(**kwargs) -> tuple[FlatTransform, MaskedEncoderWrapper]:
    transform = FlatTransform()
    model = models_mae.MaskedAutoencoderViT.from_pretrained("medarc/fm_mae_vit_base_patch16-16.hcp")
    model = MaskedEncoderWrapper(model.encoder)
    return transform, model


@register_model
def flat_mae_base_patch16_2(**kwargs) -> tuple[FlatTransform, MaskedEncoderWrapper]:
    transform = FlatTransform()
    model = models_mae.MaskedAutoencoderViT.from_pretrained("medarc/fm_mae_vit_base_patch16-2.hcp")
    model = MaskedEncoderWrapper(model.encoder)
    return transform, model
