import torch.nn as nn
from torch import Tensor
from einops import rearrange

from brainmarks.models.base import Embeddings
from brainmarks.models.registry import register_model

import cortex_mae.models_mae as models_mae
from cortex_mae.inference import CortexMAE, Transform, pad_unfold


class CortexMAEWrapper(nn.Module):
    __space__: str = "flat"

    def __init__(self, encoder: models_mae.MaskedEncoder):
        super().__init__()
        self.encoder = encoder
        self.num_frames = self.encoder.patchify.img_size[0]

    def forward(self, batch: dict[str, Tensor]) -> Embeddings:
        bold = batch["bold"]
        mask = batch["mask"]
        B, C, T, H, W = bold.shape
        if mask.ndim == 3:
            mask = mask[:, None, None, :, :]
        mask = mask.expand_as(bold)

        # pad/truncate and unfold into non-overlapping sliding windows
        bold, mask, num_clips = pad_unfold(bold, mask, num_frames=self.num_frames)

        cls_embeds, reg_embeds, patch_embeds = self.encoder.forward_embedding(bold, mask)

        # unflatten batch and clip dimensions
        if cls_embeds is not None:
            cls_embeds = rearrange(cls_embeds, "(b n) l d -> b (n l) d", n=num_clips)
            cls_embeds = cls_embeds.mean(dim=1, keepdim=True)
        if reg_embeds is not None:
            reg_embeds = rearrange(reg_embeds, "(b n) l d -> b (n l) d", n=num_clips)
        patch_embeds = rearrange(patch_embeds, "(b n) l d -> b (n l) d", n=num_clips)

        return Embeddings(cls_embeds, reg_embeds, patch_embeds)


@register_model
def cortex_mae(
    *,
    model_name: str = "cortex_mae_flat",
    ckpt_path: str | None = None,
    scratch_init: bool = False,
    keep_blocks: int | None = None,
) -> tuple[Transform, CortexMAEWrapper]:
    if ckpt_path is not None:
        model = CortexMAE.from_checkpoint(ckpt_path)
    else:
        model = CortexMAE.from_pretrained(model_name)

    input_space = model.args.input_space
    transform = model.transform
    # re-init weights to train from scratch
    if scratch_init:
        model.model.init_weights()
    # remove some vit blocks (nb keep_blocks=0 is patch embed only)
    encoder = model.model.encoder
    if keep_blocks is not None:
        encoder.blocks = encoder.blocks[:keep_blocks]
    model = CortexMAEWrapper(encoder)
    model.__space__ = input_space
    return transform, model
