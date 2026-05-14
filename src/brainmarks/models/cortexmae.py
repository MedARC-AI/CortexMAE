import torch.nn as nn
from torch import Tensor

from brainmarks.models.base import Embeddings
from brainmarks.models.registry import register_model

import cortex_mae.models_mae as models_mae
from cortex_mae.inference import CortexMAE, Transform, forward_embedding, unpack_batch


class CortexMAEWrapper(nn.Module):
    __space__: str = "flat"

    def __init__(self, encoder: models_mae.MaskedEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, batch: dict[str, Tensor]) -> Embeddings:
        bold, mask = unpack_batch(batch)
        cls_embeds, reg_embeds, patch_embeds = forward_embedding(self.encoder, bold, mask)

        # embeddings are shape [B N L D], where N = number of sliding windows; flatten
        B, N, L, D = patch_embeds.shape
        if cls_embeds is not None:
            cls_embeds = cls_embeds.mean(dim=1)
        if reg_embeds is not None:
            reg_embeds = reg_embeds.flatten(1, 2)
        patch_embeds = patch_embeds.flatten(1, 2)

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
