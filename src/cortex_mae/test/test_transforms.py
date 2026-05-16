import torch

from cortex_mae.transforms import MNICortexUnmask


def test_mni_cortex_unmask():
    unmask = MNICortexUnmask()
    T, D = 16, unmask.dim
    bold = torch.randn(T, D)

    # zero out voxels not covered by patches
    bold = bold * torch.as_tensor(unmask.restore_mask)

    # forward
    sample = unmask({"bold": bold})
    assert sample["bold"].shape == (1, T, unmask.num_patches, unmask.patch_dim)
    assert sample["mask"].shape == (unmask.num_patches, unmask.patch_dim)
    assert sample["mask"].dtype == torch.bool

    # inverse
    bold_restored = unmask.inverse(sample["bold"].squeeze(0))
    assert bold_restored.shape == (T, D)

    # round trip preserves values
    torch.testing.assert_close(bold_restored, bold)
