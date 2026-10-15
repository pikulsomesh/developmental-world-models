"""A shared latent toy world rendered into three senses (vision, audition, touch).

The point of this file is the fairness trick for cross-sense comparison: every
sense sees the *same* latent world (an object of identity k moving along a 1-D
track and bouncing off walls), and every violation-of-expectation (VoE) event is
applied to the latent state *before* rendering. Differences between senses in
how well a model tracks the object, or how surprised it is by a violation, can
then be attributed to the sense's input format and developmental constraints,
not to different tasks.

Each sense also has a natural partial-observability region, analogous to an
occluder:
  vision : an occluder hides the object for x in [0.55, 0.80]
  touch  : the object is only felt when within reach, x < 0.5
  audio  : always audible, but position is only coded by loudness
"""
from __future__ import annotations

import numpy as np

SENSES = ("vision", "audio", "touch")
N_ID = 4
V_SIDE = 12          # vision is V_SIDE x V_SIDE pixels
A_BINS = 24          # audio frequency bins (low -> high)
T_TAXELS = 16        # taxels along a 1-D "arm"
DIMS = {"vision": V_SIDE * V_SIDE, "audio": A_BINS, "touch": T_TAXELS}

OCC_LO, OCC_HI = 0.55, 0.80
REACH = 0.5

# 3x3 shape per identity for vision
_SHAPES = np.array([
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],   # plus
    [[1, 0, 1], [0, 1, 0], [1, 0, 1]],   # cross
    [[1, 1, 1], [1, 0, 1], [1, 1, 1]],   # ring
    [[1, 1, 0], [1, 1, 0], [0, 0, 1]],   # odd
], dtype=np.float32)
# fundamental audio bin per identity; harmonics at 2x, 3x
_PITCH = np.array([2.0, 3.0, 4.5, 6.5])
# temporal vibration period per identity for touch (texture)
_TEX_PERIOD = np.array([2, 3, 4, 6])


def sample_latents(B, T, rng, violation=None, t_v=None, ids=None):
    """Return dict with x (B,T) position, k (B,T) identity, hit (B,T) wall collision.

    violation: None | "teleport" (position jumps by 0.4 at t_v) |
               "identity" (identity changes at t_v).
    The RNG stream is consumed identically with and without a violation, so
    calling twice with the same seed gives matched control/violation pairs.
    ids: optional pair (a, b) of identities; objects are drawn from it and an
    identity violation swaps a <-> b (used for the perceptual-narrowing test).
    """
    x = rng.uniform(0.05, 0.95, B)
    v = rng.choice([-1.0, 1.0], B) * rng.uniform(0.03, 0.06, B)
    k = rng.integers(0, N_ID, B)
    k_alt = (k + rng.integers(1, N_ID, B)) % N_ID
    if ids is not None:
        pick = k % 2
        k = np.where(pick == 0, ids[0], ids[1])
        k_alt = np.where(pick == 0, ids[1], ids[0])
    X = np.zeros((B, T), np.float32)
    K = np.zeros((B, T), np.int64)
    H = np.zeros((B, T), np.float32)
    for t in range(T):
        if violation == "teleport" and t == t_v:
            x = (x + 0.4) % 1.0
        if violation == "identity" and t == t_v:
            k = k_alt
        x = x + v
        lo, hi = x < 0, x > 1
        x = np.where(lo, -x, np.where(hi, 2 - x, x))
        v = np.where(lo | hi, -v, v)
        X[:, t], K[:, t], H[:, t] = x, k, (lo | hi)
    return {"x": X, "k": K, "hit": H}


def _gauss1d(n, centers, width):
    idx = np.arange(n, dtype=np.float32)
    return np.exp(-0.5 * ((idx[None, :] - centers[:, None]) / width) ** 2)


def render(sense, lat):
    x, k, hit = lat["x"], lat["k"], lat["hit"]
    B, T = x.shape
    if sense == "vision":
        img = np.zeros((B, T, V_SIDE, V_SIDE), np.float32)
        col = np.clip(np.round(x * (V_SIDE - 3)).astype(int), 0, V_SIDE - 3)
        r0 = V_SIDE // 2 - 1
        for b in range(B):
            for t in range(T):
                img[b, t, r0:r0 + 3, col[b, t]:col[b, t] + 3] = _SHAPES[k[b, t]]
        c_lo, c_hi = int(OCC_LO * (V_SIDE - 1)), int(np.ceil(OCC_HI * (V_SIDE - 1)))
        img[:, :, r0 - 2:r0 + 5, c_lo:c_hi + 1] = 0.5      # grey occluder
        return img.reshape(B, T, -1)
    if sense == "audio":
        f0 = _PITCH[k]                                      # (B,T)
        loud = 0.25 + 0.75 * (1.0 - x)                      # listener at x=0
        spec = np.zeros((B, T, A_BINS), np.float32)
        for h, amp in ((1, 1.0), (2, 0.6), (3, 0.35)):
            c = (f0 * h).reshape(-1)
            spec += amp * _gauss1d(A_BINS, c, 0.6).reshape(B, T, A_BINS)
        spec *= loud[..., None]
        spec += 0.6 * hit[..., None]                        # broadband click on collision
        return spec.astype(np.float32)
    if sense == "touch":
        pos = (x / REACH) * (T_TAXELS - 1)
        tex = _TEX_PERIOD[k]
        phase = (np.arange(T)[None, :] % tex) < (tex / 2)   # vibration on/off
        amp = np.where(x < REACH, 0.6 + 0.4 * phase, 0.0)
        p = _gauss1d(T_TAXELS, pos.reshape(-1), 0.8).reshape(B, T, T_TAXELS)
        return (p * amp[..., None]).astype(np.float32)
    raise ValueError(sense)


def unobserved_mask(sense, lat):
    """True where the object is hidden for this sense (the 'occluded' timesteps)."""
    x = lat["x"]
    if sense == "vision":
        col = np.clip(np.round(x * (V_SIDE - 3)).astype(int), 0, V_SIDE - 3)
        c_lo, c_hi = int(OCC_LO * (V_SIDE - 1)), int(np.ceil(OCC_HI * (V_SIDE - 1)))
        return (col >= c_lo) & (col + 2 <= c_hi)
    if sense == "touch":
        return x >= REACH
    return np.zeros_like(x, dtype=bool)
