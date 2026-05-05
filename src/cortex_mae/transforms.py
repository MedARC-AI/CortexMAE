import random
from functools import cache
from typing import Literal

import torch
import torchvision.transforms.v2 as v2
import torchvision.tv_tensors as tvt
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import neuromaps.transforms

import cortex_mae.nisc as nisc

# shared flat map resampler
_FLAT_RESAMPLER = nisc.flat_resampler_fslr64k_224_560()


class ToTensor:
    def __call__(self, sample: dict) -> dict:
        bold = sample["bold"]
        bold = torch.as_tensor(bold, dtype=torch.float32)
        return {**sample, "bold": bold}

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class CoordUnNormalize:
    def __call__(self, sample: dict) -> dict:
        bold = sample["bold"]
        mean = torch.as_tensor(sample["mean"], dtype=torch.float32)
        std = torch.as_tensor(sample["std"], dtype=torch.float32)
        bold = bold * std + mean
        bold = (bold - bold.mean()) / bold.std()
        return {**sample, "bold": bold}

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Normalize:
    def __init__(self, mode: Literal["frame", "global"], eps: float = 1e-6):
        self.mode = mode
        self.eps = eps

    def __call__(self, sample: dict) -> dict:
        bold = sample["bold"]
        T, D = bold.shape
        dim = {"global": None, "frame": 1}[self.mode]
        mean = bold.mean(dim=dim, keepdim=True)
        std = bold.std(dim=dim, keepdim=True)
        bold = (bold - mean) / (std + self.eps)
        return {**sample, "bold": bold}

    def __repr__(self):
        return f"{self.__class__.__name__}(mode='{self.mode}')"


class TemporalRandomResizedCrop:
    """
    Random temporal crop and resize, to jitter temporal resolution.
    scale is a scaling factor where 0.5 means the effective tr is between 0.5x and 2x
    the original tr.
    """

    def __init__(self, scale: float = 0.8, num_frames: int = 16):
        assert 0 < scale < 1
        self.scale = scale
        self.num_frames = num_frames

    def __call__(self, sample: dict) -> dict:
        bold = sample["bold"]
        T, D = bold.shape
        min_t = round(self.num_frames * self.scale)
        max_t = round(self.num_frames / self.scale)
        assert T >= max_t, (
            f"invalid clip length {T} for temporal scale {self.scale} and num frames {self.num_frames}"
        )

        t = random.randint(min_t, max_t)  # nb endpoints included
        start = random.randint(0, T - t)

        bold = bold[start : start + t]
        bold = F.interpolate(
            bold.T.unsqueeze(0),
            size=self.num_frames,
            mode="linear",
        )  # [1, D, T]
        bold = bold.squeeze(0).T.contiguous()
        return {**sample, "bold": bold}

    def __repr__(self):
        return f"{self.__class__.__name__}(scale={self.scale}, num_frames={self.num_frames})"


class TemporalCenterCrop:
    def __init__(self, num_frames: int = 16):
        self.num_frames = num_frames

    def __call__(self, sample: dict) -> dict:
        bold = sample["bold"]
        T, D = bold.shape
        assert T >= self.num_frames, f"clip too short {T} for num frames {self.num_frames}"
        start = (T - self.num_frames) // 2
        bold = bold[start : start + self.num_frames]
        return {**sample, "bold": bold}

    def __repr__(self):
        return f"{self.__class__.__name__}(num_frames={self.num_frames})"


class FlatUnmask:
    """
    The unmasking functions take flattened (raveled) vector time, shape (T, D), and
    produce unmasked time series, shape (C, T, H, W).
    """

    dim = 77763

    def __init__(self):
        self.mask = torch.as_tensor(_FLAT_RESAMPLER.mask_)

    def __call__(self, sample: dict) -> dict:
        bold = sample["bold"]
        T, D = bold.shape
        assert D == self.dim, f"input dim {D} doesn't match expected {self.dim}"
        bold_ = torch.zeros(1, T, *self.mask.shape)
        bold_[..., self.mask] = bold
        bold = bold_
        mask = self.mask
        return {**sample, "bold": bold, "mask": mask}

    def to_flat(self, bold: torch.Tensor) -> torch.Tensor:
        # just for consistency
        return bold

    def __repr__(self):
        return f"{self.__class__.__name__}({tuple(self.mask.shape)})"


class ParcelUnmask:
    def __init__(self, parc: np.ndarray, dim: int):
        assert parc.ndim == 1
        assert parc.shape[0] == nisc.FSLR64K_NUM_VERTICES
        self.parc = torch.as_tensor(parc, dtype=torch.int64)
        self.dim = dim

    def __call__(self, sample: dict) -> dict:
        bold = sample["bold"]
        T, D = bold.shape
        assert D == self.dim, f"input dim {D} doesn't match expected {self.dim}"
        bold = bold[None, :, :, None]  # [1, T, D, 1]
        mask = torch.ones(D, 1, dtype=torch.bool)  # [D, 1] spatial mask only
        return {**sample, "bold": bold, "mask": mask}

    def to_flat(self, bold: torch.Tensor) -> torch.Tensor:
        # map parcellated values to flat map for visualization
        # patches [N, P] -> vector [D]
        *shape, D, _ = bold.shape
        assert D == self.dim, f"input dim {D} doesn't match expected {self.dim}"
        bold = bold.squeeze(-1)
        # vector [D] -> surface [V]
        (V,) = self.parc.shape
        # parcellation codes 0 as background and 1-indexed rois
        parc_mask = self.parc > 0
        parc_ids = self.parc - 1
        bold = bold[..., parc_ids] * parc_mask
        # surface [V] -> flat [H, W]
        bold = _FLAT_RESAMPLER.transform(bold.numpy(), interpolation="nearest")
        bold = torch.as_tensor(bold)
        return bold

    def __repr__(self):
        return f"{self.__class__.__name__}({self.dim})"


class Schaefer400Unmask(ParcelUnmask):
    def __init__(self):
        parc_path = nisc.fetch_schaefer(400, space="fslr64k")
        parc = nisc.read_cifti_surf_data(parc_path).squeeze(0)
        super().__init__(parc, dim=400)


class Schaefer400TianS3Unmask(ParcelUnmask):
    def __init__(self):
        parc_path = nisc.fetch_schaefer_tian(400, 3, space="fslr91k")
        # nb, we only pass the surface part of the parcellation, used for vis only
        parc = nisc.read_cifti_surf_data(parc_path).squeeze(0)
        super().__init__(parc, dim=450)


class A424Unmask(ParcelUnmask):
    def __init__(self):
        parc_path = nisc.fetch_a424(cifti=True)
        # nb, we only pass the surface part of the parcellation, used for vis only
        parc = nisc.read_cifti_surf_data(parc_path).squeeze(0)
        super().__init__(parc, dim=424)


class MNICortexUnmask:
    """
    unmasks mni cortex data into patchified format.
    input: (T, D) where D=132032 -> output: bold (1, T, N, P), mask (N, P)
    """

    dim = 132032

    def __init__(self, patch_size: int = 8, threshold: float = 0.10):
        self.patch_size = patch_size
        self.threshold = threshold

        # load cortex mask from schaefer400 parcellation
        # important, mask must be in RAS orientation
        mask_path = nisc.fetch_schaefer(400, space="mni")
        mask_img = nib.load(mask_path)
        mask_img = nisc.ensure_mni152_2mm_ras(mask_img, interpolation="nearest")
        mask = np.ascontiguousarray(mask_img.get_fdata().T) > 0  # (Z, Y, X)
        assert mask.sum() == self.dim

        # gather_ids: (N, P) array of indices in [0, D) into the original data
        # patch_mask: (N, P) mask of non-background data in patchified space
        gather_ids, patch_mask = _make_volume_patch_gather_ids(
            mask, patch_size=patch_size, threshold=threshold
        )

        self.mask_img: nib.Nifti1Image = mask_img
        self.mask = torch.as_tensor(mask)
        self.gather_ids = torch.as_tensor(gather_ids)
        self.patch_mask = torch.as_tensor(patch_mask)
        self.num_patches, self.patch_dim = self.patch_mask.shape

        # indices to apply inverse transform
        self.restore_ids = torch.argsort(self.gather_ids[self.patch_mask])
        self.restore_mask = np.zeros(self.dim, dtype=bool)
        self.restore_mask[self.gather_ids[self.patch_mask]] = True

    def __call__(self, sample: dict) -> dict:
        bold = sample["bold"]
        bold = self.transform(bold)
        bold = bold[None]  # (1, T, N, P)
        return {**sample, "bold": bold, "mask": self.patch_mask}

    def transform(self, bold: torch.Tensor) -> torch.Tensor:
        *shape, D = bold.shape
        assert D == self.dim, f"input dim {D} doesn't match expected {self.dim}"

        bold = bold[..., self.gather_ids] * self.patch_mask  # (T, N, P)
        return bold

    def inverse(self, bold: torch.Tensor) -> torch.Tensor:
        *shape, N, P = bold.shape
        values = bold[..., self.patch_mask]
        values = values[..., self.restore_ids]
        # nb, we assume we're on cpu with no grad
        bold = torch.zeros((*shape, self.dim), dtype=bold.dtype)
        bold[..., self.restore_mask] = values
        return bold

    def to_flat(self, bold: torch.Tensor) -> torch.Tensor:
        # map patchified cortex values to flat map for visualization
        # patches [N, P] -> vector [D]
        bold = self.inverse(bold)
        *shape, D = bold.shape
        bold = bold.reshape(-1, D)
        N, _ = bold.shape
        # vector [D] -> volume [Z, Y, X]
        bold_ = torch.zeros((N, *self.mask.shape), dtype=bold.dtype)
        bold_[:, self.mask] = bold
        bold = bold_
        # volume [Z, Y, X] -> surface [V] with neuromaps
        # nifti doesn't support bool
        bold = bold.numpy().T  # [Z, Y, X] -> [X, Y, Z] F order
        is_bool = np.issubdtype(bold.dtype, np.bool_)
        if is_bool:
            bold = bold.astype(np.float32)
        bold_img = nib.Nifti1Image(bold, affine=self.mask_img.affine)
        bold_lh, bold_rh = neuromaps.transforms.mni152_to_fslr(bold_img, method="nearest")
        bold = np.stack(
            [
                np.concatenate([row_lh.data, row_rh.data])
                for row_lh, row_rh in zip(bold_lh.darrays, bold_rh.darrays)
            ]
        )
        # surface [V] -> flat [H, W]
        bold = _FLAT_RESAMPLER.transform(bold, interpolation="nearest")
        if is_bool:
            bold = bold > 0
        _, H, W = bold.shape
        bold = bold.reshape((*shape, H, W))
        bold = torch.as_tensor(bold)
        return bold

    def __repr__(self):
        s = (
            f"{self.num_patches}, {self.patch_dim}, "
            f"patch_size={self.patch_size}, threshold={self.threshold}"
        )
        return f"{self.__class__.__name__}({s})"


def _make_volume_patch_gather_ids(mask: np.ndarray, patch_size: int = 8, threshold: float = 0.25):
    p = patch_size

    # pad to divisible by patch_size
    mask_pad = np.pad(mask, [(0, -d % p) for d in mask.shape])

    # index mapping: position -> index into masked array, or -1 for background
    ids = np.full(mask_pad.shape, -1, dtype=np.int64)
    ids[mask_pad] = np.arange(mask_pad.sum())

    # rearrange into 3D patches: (D, H, W) -> (N_total, P)
    mask_patches = _to_patches(mask_pad, p)
    ids_patches = _to_patches(ids, p)

    # keep patches with sufficient brain coverage
    keep = mask_patches.mean(axis=1) > threshold

    gather_ids = ids_patches[keep]  # (N, P)
    patch_mask = mask_patches[keep]
    return gather_ids, patch_mask


def _to_patches(x: np.ndarray, p: int):
    D, H, W = x.shape
    x = x.reshape(D // p, p, H // p, p, W // p, p)
    return x.transpose(0, 2, 4, 1, 3, 5).reshape(-1, p**3)


class FlatRandomResizedCrop:
    """
    expected flat map image size (224, 560)
    some defaults:
        weak: scale=(0.8, 1.0), ratio=(2.5, 2.5) (~1 patch cropping and no aspect change)
        moderate: scale=(0.25, 1.0), ratio=(2.0, 3.125) (up to 50% crop per side and 80%
        aspect change)
    """

    img_size = (224, 560)

    def __init__(
        self,
        crop_scale: float = 0.8,
        crop_aspect: float = 1.0,
        interpolation: v2.InterpolationMode = v2.InterpolationMode.BICUBIC,
    ):
        assert 0 < crop_scale < 1.0, f"invalid {crop_scale=}"
        assert 0 < crop_aspect <= 1.0, f"invalid {crop_aspect=}"

        self.crop_scale = crop_scale
        self.crop_aspect = crop_aspect

        scale = (crop_scale, 1.0)
        H, W = self.img_size
        aspect = W / H
        ratio = (aspect * crop_aspect, aspect / crop_aspect)

        self.transform = v2.RandomResizedCrop(
            size=self.img_size,
            scale=scale,
            ratio=ratio,
            interpolation=interpolation,
        )

    def __call__(self, sample):
        bold = sample["bold"]
        mask = sample["mask"]
        C, T, H, W = bold.shape
        bold = bold.reshape(-1, H, W)

        bold, mask = self.transform(bold, tvt.Mask(mask))
        _, H, W = bold.shape
        bold = bold.reshape(C, T, H, W)
        return {**sample, "bold": bold, "mask": mask}

    def __repr__(self):
        c = self.__class__.__name__
        return f"{c}(crop_scale={self.crop_scale}, crop_aspect={self.crop_aspect})"


class Clip:
    def __init__(self, vmax: float | None = None):
        self.vmax = vmax

    def __call__(self, sample):
        bold = sample["bold"]
        if self.vmax is not None and self.vmax > 0:
            bold = torch.clamp(bold, min=-self.vmax, max=self.vmax)
        return {**sample, "bold": bold}

    def __repr__(self):
        c = self.__class__.__name__
        return f"{c}(vmax={self.vmax})"


class GrayJitter:
    def __init__(self, brightness: float | None = None, contrast: float | None = None):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, sample):
        bold = sample["bold"]
        mask = sample["mask"]

        if self.brightness is not None:
            brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
            bold = bold * brightness_factor
        if self.contrast is not None:
            contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            mean = (mask * bold).sum() / mask.expand_as(bold).sum()
            bold = (bold - mean) * contrast_factor + mean
            bold = bold * mask

        return {**sample, "bold": bold}

    def __repr__(self):
        c = self.__class__.__name__
        return f"{c}(brightness={self.brightness}, contrast={self.contrast})"


def make_transform(
    space: Literal["flat", "schaefer400", "mni_cortex", "schaefer400_tians3", "a424"] = "flat",
    num_frames: int = 16,
    normalize: Literal["global", "frame"] | None = "frame",
    clip_vmax: float | None = 3.0,
    tr_scale: float | None = None,
    crop_scale: float | None = None,
    crop_aspect: float | None = None,
    gray_jitter: float | None = None,
    no_coord_normalize: bool = False,
) -> v2.Compose:
    assert crop_scale is None or space == "flat", "crop only supported for flat maps"

    transforms = [ToTensor()]

    if no_coord_normalize:
        transforms.append(CoordUnNormalize())

    if tr_scale and tr_scale < 1:
        transforms.append(TemporalRandomResizedCrop(scale=tr_scale, num_frames=num_frames))
    else:
        transforms.append(TemporalCenterCrop(num_frames=num_frames))

    if normalize:
        transforms.append(Normalize(normalize))
    if clip_vmax and clip_vmax > 0:
        transforms.append(Clip(clip_vmax))

    unmask = get_unmask(space)
    transforms.append(unmask)

    if crop_scale and crop_scale < 1:
        transforms.append(FlatRandomResizedCrop(crop_scale, crop_aspect or 1.0))

    if gray_jitter and gray_jitter > 0:
        transforms.append(GrayJitter(gray_jitter, gray_jitter))

    return v2.Compose(transforms)


@cache
def get_unmask(
    space: Literal["flat", "schaefer400", "mni_cortex", "schaefer400_tians3", "a424"] = "flat",
):
    """
    return singleton unmask fn.

    (not sure if this is the best way to do this but ok. I just need access to the
    `to_flat` function in some places.)
    """
    unmask_cls = {
        "flat": FlatUnmask,
        "schaefer400": Schaefer400Unmask,
        "mni_cortex": MNICortexUnmask,
        "schaefer400_tians3": Schaefer400TianS3Unmask,
        "a424": A424Unmask,
    }[space]
    unmask = unmask_cls()
    return unmask
