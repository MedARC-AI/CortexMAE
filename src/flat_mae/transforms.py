import random
from typing import Literal

import torch
import torchvision.transforms.v2 as v2
import torchvision.tv_tensors as tvt
import torch.nn.functional as F

import fmri_fm_eval.nisc as nisc

# TODO:
#   - pca noising, ie pink noise


class ToTensor:
    def __call__(self, sample: dict) -> dict:
        bold = sample["bold"]
        bold = torch.as_tensor(bold, dtype=torch.float32)
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
        return f"{self.__class__.__name__}(mode={self.mode})"


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
        assert min_t <= T <= max_t, (
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

    def __init__(self):
        resampler = nisc.flat_resampler_fslr64k_224_560()
        self.mask = torch.as_tensor(resampler.mask_)

    def __call__(self, sample: dict) -> dict:
        bold = sample["bold"]
        T, D = bold.shape
        bold_ = torch.zeros(1, T, *self.mask.shape)
        bold_[..., self.mask] = bold
        bold = bold_
        mask = self.mask
        return {**sample, "bold": bold, "mask": mask}

    def __repr__(self):
        return f"{self.__class__.__name__}({tuple(self.mask.shape)})"


class ParcelUnmask:
    def __call__(self, sample: dict) -> dict:
        bold = sample["bold"]
        T, D = bold.shape
        bold = bold[None, :, :, None]  # [1, T, D, 1]
        mask = torch.ones(D, 1, dtype=torch.bool)  # [D, 1] spatial mask only
        return {**sample, "bold": bold, "mask": mask}

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class MNICortexUnmask:
    def __call__(self, sample: dict) -> dict:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}()"


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
        **kwargs,
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
            **kwargs,
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
            mean = bold.sum() / mask.expand_as(bold).sum()
            bold = (bold - mean) * contrast_factor + mean
            bold = bold * mask

        return {**sample, "bold": bold}

    def __repr__(self):
        c = self.__class__.__name__
        return f"{c}(brightness={self.brightness}, contrast={self.contrast})"


class GaussianJitter:
    def __init__(self, std: float = 1.0):
        self.std = std

    def __call__(self, sample):
        bold = sample["bold"]
        mask = sample["mask"]

        if self.std > 0:
            bold = bold + self.std * torch.randn_like(bold)
            bold = bold * mask

        return {**sample, "bold": bold}

    def __repr__(self):
        c = self.__class__.__name__
        return f"{c}({self.std})"


class Transform:
    def __init__(
        self,
        space: Literal["flat", "schaefer400", "mni_cortex"] = "flat",
        num_frames: int = 16,
        normalize: Literal["global", "frame"] | None = None,
        clip_vmax: float | None = 3.0,
        tr_scale: float | None = None,
        crop_scale: float | None = None,
        crop_aspect: float | None = None,
        gray_jitter: float | None = None,
        gauss_sigma: float | None = None,
    ):
        assert crop_scale is None or space == "flat", "crop only supported for flat maps"

        transforms = [ToTensor()]

        if tr_scale and tr_scale < 1:
            transforms.append(TemporalRandomResizedCrop(scale=tr_scale, num_frames=num_frames))
        else:
            transforms.append(TemporalCenterCrop(num_frames=num_frames))

        if normalize:
            transforms.append(Normalize(normalize))
        if clip_vmax and clip_vmax > 0:
            transforms.append(Clip(clip_vmax))

        unmask_cls = {
            "flat": FlatUnmask,
            "schaefer400": ParcelUnmask,
            "mni_cortex": MNICortexUnmask,
        }[space]
        transforms.append(unmask_cls())

        if crop_scale and crop_scale < 1:
            transforms.append(FlatRandomResizedCrop(crop_scale, crop_aspect or 1.0))

        if gray_jitter and gray_jitter > 0:
            transforms.append(GrayJitter(gray_jitter, gray_jitter))

        # extra noise transforms applied only to input images, not targets
        noise_transforms = []
        if gauss_sigma and gauss_sigma > 0:
            noise_transforms.append(GaussianJitter(gauss_sigma))

        self.transform = v2.Compose(transforms)

        if noise_transforms:
            self.noise_transform = v2.Compose(noise_transforms)
        else:
            self.noise_transform = None

    def __call__(self, sample):
        sample = self.transform(sample)
        if self.noise_transform is not None:
            sample["bold_clean"] = sample["bold"]
            sample = self.noise_transform(sample)
        return sample

    def __repr__(self):
        c = self.__class__.__name__
        s = f"    transform={self.transform},\n    noise_transform={self.noise_transform}"
        s = f"{c}(\n{s}\n)"
        return s
