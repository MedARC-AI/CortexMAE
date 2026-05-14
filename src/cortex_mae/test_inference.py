import os

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from cortex_mae.inference import (
    CortexMAE,
    EmbeddingOutput,
    ReconstructionOutput,
    resolve_file,
)
from cortex_mae.transforms import FlatUnmask


def _flat_args(**overrides):
    args = OmegaConf.create(
        {
            "input_space": "flat",
            "img_size": (224, 560),
            "in_chans": 1,
            "patch_size": 16,
            "num_frames": 16,
            "t_patch_size": 4,
            "masking": "tube",
            "mask_ratio": 0.9,
            "model": "mae_vit_base",
            "model_kwargs": {},
            "normalize": "frame",
            "clip_vmax": 3.0,
        }
    )
    return OmegaConf.merge(args, overrides)


def _parcel_args(**overrides):
    args = OmegaConf.create(
        {
            "input_space": "schaefer400",
            "img_size": (400, 1),
            "in_chans": 1,
            "patch_size": 1,
            "num_frames": 16,
            "t_patch_size": 4,
            "masking": "tube",
            "mask_ratio": 0.9,
            "model": "mae_vit_base",
            "model_kwargs": {},
            "normalize": "frame",
            "clip_vmax": 3.0,
        }
    )
    return OmegaConf.merge(args, overrides)


def _dummy_sample(T: int = 16, D: int = FlatUnmask.dim):
    rng = np.random.default_rng(0)
    series = rng.standard_normal((T, D)).astype(np.float32)
    return {
        "bold": series,
        "mean": np.zeros(D, dtype=np.float32),
        "std": np.ones(D, dtype=np.float32),
        "tr": 1.0,
    }


def test_from_config_constructs_flat():
    model = CortexMAE.from_config(_flat_args())
    assert model.model is not None
    assert model.reader is not None
    assert model.transform is not None
    assert model.mask_fn is not None


def test_from_config_constructs_schaefer400():
    model = CortexMAE.from_config(_parcel_args())
    assert model.transform.unmask.__class__.__name__ == "Schaefer400Unmask"


def test_forward_masked_recon_smoke():
    model = CortexMAE.from_config(_flat_args())
    sample = _dummy_sample()
    out = model.run_masked_recon(sample)
    assert isinstance(out, ReconstructionOutput)
    assert out.images.shape == out.pred_images.shape
    assert out.images.ndim == 6  # [B, C, N, T, H, W]


def test_forward_embedding_smoke():
    model = CortexMAE.from_config(_flat_args())
    sample = _dummy_sample()
    out = model.run_embedding(sample)
    assert isinstance(out, EmbeddingOutput)
    assert out.patch_embeds.ndim == 4
    B, N, L, D = out.patch_embeds.shape
    assert B == 1 and D == 768


def test_from_checkpoint(tmp_path):
    src = CortexMAE.from_config(_parcel_args())
    ckpt_path = tmp_path / "ckpt.pth"
    torch.save(
        {
            "args": OmegaConf.to_container(src.args, resolve=True),
            "model": src.model.state_dict(),
        },
        ckpt_path,
    )
    loaded = CortexMAE.from_checkpoint(str(ckpt_path))
    # weights round-trip
    src_state = src.model.state_dict()
    loaded_state = loaded.model.state_dict()
    for k, v in src_state.items():
        assert torch.equal(v, loaded_state[k])


@pytest.mark.parametrize(
    "path",
    [
        "s3://openneuro.org/ds006072/dataset_description.json",
        "https://github.com/ThomasYeoLab/CBIG/raw/refs/heads/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/HCP/fslr32k/cifti/Schaefer2018_1000Parcels_17Networks_order_info.txt",
    ],
)
def test_resolve_file(path, tmp_path):
    kwargs = {}
    if path.startswith("s3://"):
        kwargs["anon"] = True
    local_path = resolve_file(path, cache_dir=tmp_path, **kwargs)
    local_path = resolve_file(path, cache_dir=tmp_path)
    local_path = resolve_file(local_path, cache_dir=tmp_path)
    assert local_path.exists()


@pytest.mark.skipif(
    os.environ.get("CORTEX_MAE_SLOW_TESTS") != "1",
    reason="set CORTEX_MAE_SLOW_TESTS=1 to enable network-dependent e2e test",
)
def test_pretrained_end_to_end(tmp_path):
    """Full pipeline: from_pretrained + run_masked_recon on a real file.

    Requires HuggingFace + an example file. Skipped by default; run with:
        CORTEX_MAE_SLOW_TESTS=1 pytest src/cortex_mae/test_inference.py -k pretrained
    """
    model = CortexMAE.from_pretrained("cortex_mae_flat")
    model.set_device()

    url = "s3://openneuro.org/ds006072/NON_BIDS/ciftis/sub-1_Drug2_rsfMRI_uout_bpss_sr_noGSR_sm4.dtseries.nii"
    path = resolve_file(url, cache_dir=tmp_path, anon=True)
    out = model.run_embedding(path)
    B, N, L, D = out.patch_embeds.shape
    assert B == 1 and D == 768
