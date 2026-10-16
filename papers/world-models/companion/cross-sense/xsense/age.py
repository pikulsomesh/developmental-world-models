"""Age-dependent sensory front ends ("input maturation"), one per sense.

Ages are in months relative to birth; negative ages are prenatal (third
trimester, about -3 months). Values are coarse, literature-anchored settings for
a toy world, not calibrated psychophysics. Sources to check in the paper:
  vision : acuity ~1 cpd at birth -> ~30 cpd by 12 mo; contrast sensitivity
           ~2 -> ~100 (Teller; Banks & Bennett 1988; Norcia & Tyler 1985).
           No patterned visual input before birth.
  audio  : cochlea functional ~24-26 wk gestation; womb attenuates >~500 Hz-1 kHz
           (Querleu et al. 1988); frequency resolution and thresholds mature
           over the first ~6 months (Werner 2007).
  touch  : somatosensation is the earliest-maturing sense (fetal); spatial
           acuity still coarser than adult in infancy.
Each table maps age -> (smoothing width in native units, additive noise std,
low-pass fraction for audio). Inputs at an age are what a model of that age
"perceives"; the world is the same.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, gaussian_filter

from .world import V_SIDE

AGES = (-3, 0, 3, 6, 12)

# vision: blur sigma in pixels (from acuity), noise std (from 1/contrast sensitivity)
_VISION = {-3: None, 0: (2.2, 0.25), 3: (0.9, 0.06), 6: (0.35, 0.02), 12: (0.0, 0.01)}
# audio: frequency smear sigma (bins), noise std, fraction of bins passed (low-pass)
_AUDIO = {-3: (1.5, 0.10, 0.30), 0: (1.2, 0.08, 1.0), 3: (0.8, 0.05, 1.0),
          6: (0.4, 0.03, 1.0), 12: (0.0, 0.02, 1.0)}
# touch: spatial smear sigma (taxels), noise std
_TOUCH = {-3: (1.6, 0.06), 0: (1.2, 0.05), 3: (0.9, 0.04), 6: (0.6, 0.03), 12: (0.3, 0.02)}


def available(sense, age):
    return not (sense == "vision" and age < 0)


def perceive(sense, X, age, rng, noise_on=True):
    """Apply the age's front end to rendered input X of shape (B, T, D)."""
    X = X.copy()
    if sense == "vision":
        p = _VISION[age]
        if p is None:                       # prenatal: no patterned light
            return np.zeros_like(X) + 0.1 + 0.02 * noise_on * rng.standard_normal(X.shape).astype(np.float32)
        sig, noise = p
        if sig > 0:
            B, T, _ = X.shape
            im = X.reshape(B, T, V_SIDE, V_SIDE)
            im = gaussian_filter(im, sigma=(0, 0, sig, sig), mode="nearest")
            X = im.reshape(B, T, -1)
    elif sense == "audio":
        sig, noise, lp = _AUDIO[age]
        if sig > 0:
            X = gaussian_filter1d(X, sig, axis=-1, mode="nearest")
        if lp < 1.0:
            D = X.shape[-1]
            cut = int(lp * D)
            roll = np.ones(D, np.float32)
            roll[cut:] = np.exp(-(np.arange(D - cut)) / 1.5) * 0.1   # steep womb roll-off
            X = X * roll
    elif sense == "touch":
        sig, noise = _TOUCH[age]
        if sig > 0:
            X = gaussian_filter1d(X, sig, axis=-1, mode="nearest")
    else:
        raise ValueError(sense)
    if noise_on:
        X = X + noise * rng.standard_normal(X.shape).astype(np.float32)
    return X.astype(np.float32)
