"""Age-dependent sensory filters (non-learnable "front ends"), one per sense.

Each filter maps a clean adult-quality observation tensor to what a sensory system of
the given age would deliver. They operate on batched torch tensors so they can run
on the fly during training.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from . import capability as cap


def _gauss_kernel1d(sigma: float, device=None) -> torch.Tensor:
    if sigma < 0.05:
        return torch.ones(1, device=device)
    r = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def blur2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur over the last two dims of (N, C, H, W)."""
    k = _gauss_kernel1d(sigma, x.device)
    if k.numel() == 1:
        return x
    c = x.shape[1]
    r = k.numel() // 2
    kx = k.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    ky = k.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), kx, groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), ky, groups=c)
    return x


def blur1d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian blur over the last dim of (N, C, L)."""
    k = _gauss_kernel1d(sigma, x.device)
    if k.numel() == 1:
        return x
    c, r = x.shape[1], k.numel() // 2
    return F.conv1d(F.pad(x, (r, r), mode="replicate"), k.view(1, 1, -1).repeat(c, 1, 1), groups=c)


class MixedAge(float):
    """An age whose vision components can differ, for ablations of the front end.

    `float(MixedAge(...))` is the nominal age; `.blur`, `.contrast` and `.color` are the ages at which
    each component of the vision front end is evaluated."""

    def __new__(cls, nominal, blur=12.0, contrast=12.0, color=12.0):
        o = super().__new__(cls, nominal)
        o.blur, o.contrast, o.color = blur, contrast, color
        return o


def vision(x: torch.Tensor, age: float, gen: torch.Generator | None = None) -> torch.Tensor:
    """x: (N, 3, H, W) in [0, 1]."""
    ab, ac, ak = (age.blur, age.contrast, age.color) if isinstance(age, MixedAge) else (age, age, age)
    px = x.shape[-1]
    x = blur2d(x, cap.vision_params(ab, px)["blur_sigma_px"])
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = mean + (x - mean) * cap.vision_params(ac, px)["contrast_gain"]
    lum = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    x = lum + (x - lum) * cap.vision_params(ak, px)["color_saturation"]
    return x.clamp(0, 1)


def audio(x: torch.Tensor, age: float, gen: torch.Generator | None = None) -> torch.Tensor:
    """x: (N, 64) = 2 ears x 32 bins."""
    p = cap.audio_params(age)
    n = x.shape[0]
    x = x.view(n, 2, -1)
    bins = x.shape[-1]
    x = blur1d(x, p["freq_blur"])
    cutoff = p["hf_cutoff"] * bins
    f = torch.arange(bins, dtype=torch.float32, device=x.device)
    x = x * torch.sigmoid((cutoff - f) / 1.5)
    mono = x.mean(dim=1, keepdim=True)
    x = mono + (x - mono) * p["binaural"]
    x = x + p["noise"] * torch.randn(x.shape, generator=gen, device=x.device)
    return x.reshape(n, -1)


def touch(x: torch.Tensor, age: float, gen: torch.Generator | None = None) -> torch.Tensor:
    """x: (N, 64) = 8x8 tactile patch."""
    p = cap.touch_params(age)
    n = x.shape[0]
    x = blur2d(x.view(n, 1, 8, 8), p["blur"]).reshape(n, -1)
    return x + p["noise"] * torch.randn(x.shape, generator=gen, device=x.device)


def proprio(x: torch.Tensor, age: float, gen: torch.Generator | None = None) -> torch.Tensor:
    p = cap.proprio_params(age)
    return x + p["noise"] * torch.randn(x.shape, generator=gen, device=x.device)


FILTERS = {"vision": vision, "audio": audio, "touch": touch, "proprio": proprio}


def apply(sense: str, x: torch.Tensor, age: float, gen=None) -> torch.Tensor:
    """Apply the filter to (B, T, ...) or (N, ...) tensors."""
    shape = x.shape
    lead = 2 if (sense == "vision" and x.dim() == 5) or (sense != "vision" and x.dim() == 3) else 1
    flat = x.reshape(-1, *shape[lead:])
    return FILTERS[sense](flat, age, gen).reshape(shape)
