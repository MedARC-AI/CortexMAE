from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

from brainmarks.models.base import Embeddings
from brainmarks.models.registry import register_model

import cortex_mae.models_mae as models_mae
import cortex_mae.models_registry as models_registry
import cortex_mae.transforms as flat_transforms
from cortex_mae.models_registry import list_models  # noqa: F401


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


@register_model
def fm_mae(
    *,
    t_patch_size: int = 2,
    scratch_init: bool = False,
    keep_blocks: int | None = None,
) -> tuple[Transform, MaskedEncoderWrapper]:
    transform = Transform()
    model = models_mae.MaskedAutoencoderViT.from_pretrained(
        f"medarc/fm_mae_vit_base_patch16-{t_patch_size}.hcp"
    )
    # re-init weights to train from scratch
    if scratch_init:
        model.init_weights()
    # remove some vit blocks (nb keep_blocks=0 is patch embed only)
    if keep_blocks is not None:
        model.encoder.blocks = model.encoder.blocks[:keep_blocks]
    model = MaskedEncoderWrapper(model.encoder)
    return transform, model


@register_model
def cortex_mae(
    *,
    model_name: str = "cortex_mae_flat",
    ckpt_path: str | None = None,
    input_space: str | None = None,
    scratch_init: bool = False,
    keep_blocks: int | None = None,
) -> tuple[Transform, MaskedEncoderWrapper]:
    assert not ckpt_path or input_space, "input_space required if using a ckpt_path"

    if ckpt_path is None:
        ckpt_path = models_registry.CORTEX_MAE_MODEL_REGISTRY[model_name]
        ckpt_path = f"{models_registry.HF_PREFIX}/{ckpt_path}"
        input_space = models_registry.get_model_input_space(model_name)

    transform = Transform(space=input_space)
    model = models_mae.MaskedAutoencoderViT.from_checkpoint(ckpt_path)
    # re-init weights to train from scratch
    if scratch_init:
        model.init_weights()
    # remove some vit blocks (nb keep_blocks=0 is patch embed only)
    if keep_blocks is not None:
        model.encoder.blocks = model.encoder.blocks[:keep_blocks]
    model = MaskedEncoderWrapper(model.encoder)
    model.__space__ = input_space
    return transform, model
