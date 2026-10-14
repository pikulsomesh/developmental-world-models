"""Developmental capability tables, one per sense, plus the capacity schedule.

Each sense matures on its own timetable (heterochrony). Values are anchored to the
literature where the source proposal gives numbers (vision) and are coarse,
documented placeholders elsewhere; see docs/PLAN.md section 3 for sources and caveats.
All tables are keyed by age in months and linearly interpolated.
"""
from __future__ import annotations

import math

STAGES = [0, 3, 6, 12]


def interp(table: dict, age: float) -> float:
    ages = sorted(table)
    if age <= ages[0]:
        return float(table[ages[0]])
    if age >= ages[-1]:
        return float(table[ages[-1]])
    for a, b in zip(ages[:-1], ages[1:]):
        if a <= age <= b:
            t = (age - a) / (b - a)
            return float(table[a] * (1 - t) + table[b] * t)
    return float(table[ages[-1]])


# ---------------------------------------------------------------- vision
# Acuity (cycles/degree) and contrast sensitivity: proposal section 4.3.1
# (Norcia & Tyler 1985; Banks & Bennett 1988).
VISION_ACUITY_CPD = {0: 1.0, 1: 1.5, 2: 2.0, 3: 4.0, 4: 6.0, 5: 10.0, 6: 15.0, 9: 25.0, 12: 30.0}
VISION_CONTRAST_SENS = {0: 2, 1: 3, 2: 5, 3: 10, 4: 20, 5: 30, 6: 50, 9: 80, 12: 100}
# Proposal starts at 0.0 (fully achromatic); newborns do show some red-green
# discrimination, so we start at 0.1.
VISION_COLOR_SAT = {0: 0.1, 1: 0.2, 2: 0.4, 3: 0.7, 4: 0.9, 5: 1.0}

# ---------------------------------------------------------------- audition
# Hearing is functional before birth; frequency resolution, high-frequency
# sensitivity and binaural localisation keep maturing through the first year.
AUDIO_FREQ_BLUR_BINS = {0: 2.5, 3: 1.8, 6: 1.0, 12: 0.5}
AUDIO_HF_CUTOFF = {0: 0.6, 3: 0.75, 6: 0.9, 12: 1.0}  # fraction of the band passed
AUDIO_BINAURAL = {0: 0.3, 3: 0.5, 6: 0.8, 12: 1.0}  # 0 = mono, 1 = full L/R difference
AUDIO_NOISE = {0: 0.08, 3: 0.06, 6: 0.04, 12: 0.02}

# ---------------------------------------------------------------- touch
# Somatosensation is the earliest-maturing sense; receptive fields still shrink.
TOUCH_BLUR = {0: 1.2, 3: 0.9, 6: 0.6, 12: 0.3}
TOUCH_NOISE = {0: 0.06, 3: 0.04, 6: 0.03, 12: 0.02}

# ---------------------------------------------------------------- proprioception
PROPRIO_NOISE = {0: 0.12, 3: 0.08, 6: 0.05, 12: 0.02}

# ---------------------------------------------------------------- capacity
# Network width multiplier and working-memory window (timesteps) per stage.
WIDTH = {0: 1, 3: 2, 6: 3, 12: 4}
MEMORY_WINDOW = {0: 1, 3: 2, 6: 4, 12: 6}


def vision_params(age: float, image_px: int = 32) -> dict:
    cpd = interp(VISION_ACUITY_CPD, age)
    # Proposal: sigma = 0.5 * 30 / cpd on a ~224 px image. Rescale to our image size.
    sigma = 0.5 * (30.0 / cpd) * (image_px / 224.0)
    cs = interp(VISION_CONTRAST_SENS, age)
    # Proposal scales contrast by cs/100, which leaves a newborn 2% of all contrast.
    # Contrast sensitivity is a threshold, not a gain, so we compress on a log scale.
    contrast_gain = min(1.0, max(0.15, math.log10(cs) / 2.0))
    return {
        "acuity_cpd": cpd,
        "blur_sigma_px": sigma,
        "contrast_sensitivity": cs,
        "contrast_gain": contrast_gain,
        "color_saturation": interp(VISION_COLOR_SAT, age),
    }


def audio_params(age: float) -> dict:
    return {
        "freq_blur": interp(AUDIO_FREQ_BLUR_BINS, age),
        "hf_cutoff": interp(AUDIO_HF_CUTOFF, age),
        "binaural": interp(AUDIO_BINAURAL, age),
        "noise": interp(AUDIO_NOISE, age),
    }


def touch_params(age: float) -> dict:
    return {"blur": interp(TOUCH_BLUR, age), "noise": interp(TOUCH_NOISE, age)}


def proprio_params(age: float) -> dict:
    return {"noise": interp(PROPRIO_NOISE, age)}


def capacity(age: float) -> dict:
    stage = max(s for s in STAGES if s <= age)
    return {"width": WIDTH[stage], "window": MEMORY_WINDOW[stage]}


SENSE_PARAMS = {
    "vision": vision_params,
    "audio": audio_params,
    "touch": touch_params,
    "proprio": proprio_params,
}
