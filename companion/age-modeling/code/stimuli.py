"""Procedural stimuli for toy developmental experiments.

Vision: 8 shape classes (coarse, low spatial frequency cue) filled with 8 grating
textures (fine, high spatial frequency cue). In training the texture agrees with
the shape class with probability P_CONGRUENT, so both cues predict the label.
Test sets isolate each cue and put them in conflict (Geirhos-style shape bias).

Audio: 8 classes of synthetic "vocalisations" rendered as 32x32 log spectrograms
(time x frequency). Each class has a formant pattern; pitch, timing and noise vary.
"""
import numpy as np

S = 32
N_CLASSES = 8
P_CONGRUENT = 0.75

_yy, _xx = np.mgrid[0:S, 0:S].astype(np.float32)


def _shape_mask(cls, rng):
    cx, cy = S / 2 + rng.uniform(-4, 4), S / 2 + rng.uniform(-4, 4)
    r = rng.uniform(8, 12)
    th = rng.uniform(-0.35, 0.35)
    x, y = _xx - cx, _yy - cy
    xr = x * np.cos(th) - y * np.sin(th)
    yr = x * np.sin(th) + y * np.cos(th)
    d = np.sqrt(x ** 2 + y ** 2)
    if cls == 0:   # disk
        m = d < r
    elif cls == 1:  # square
        m = (np.abs(xr) < r * 0.8) & (np.abs(yr) < r * 0.8)
    elif cls == 2:  # triangle
        m = (yr < r * 0.7) & (yr > -r * 0.9 + 1.8 * np.abs(xr))
    elif cls == 3:  # cross
        w = r * 0.32
        m = ((np.abs(xr) < w) & (np.abs(yr) < r)) | ((np.abs(yr) < w) & (np.abs(xr) < r))
    elif cls == 4:  # ring
        m = (d < r) & (d > r * 0.55)
    elif cls == 5:  # diamond
        m = (np.abs(xr) + np.abs(yr)) < r * 1.1
    elif cls == 6:  # wide bar
        m = (np.abs(xr) < r) & (np.abs(yr) < r * 0.4)
    else:           # L shape
        w = r * 0.4
        m = ((np.abs(xr + r * 0.5) < w) & (np.abs(yr) < r)) | ((np.abs(yr - r * 0.6) < w) & (np.abs(xr) < r))
    return m.astype(np.float32)


def _texture(t, rng):
    ori = (t % 4) * np.pi / 4
    period = 3.0 if t < 4 else 4.5
    ph = rng.uniform(0, 2 * np.pi)
    u = _xx * np.cos(ori) + _yy * np.sin(ori)
    return np.sin(2 * np.pi * u / period + ph).astype(np.float32)


def render(shape_cls, tex_cls, rng, flat=False, full_texture=False):
    """One grayscale image in [0,1]. shape_cls=None with full_texture gives texture only."""
    m = np.ones((S, S), np.float32) if full_texture else _shape_mask(shape_cls, rng)
    fill = 0.65 + rng.uniform(-0.05, 0.05)
    tex = 0.0 if flat else 0.28 * _texture(tex_cls, rng)
    bg = 0.25 + rng.uniform(-0.05, 0.05)
    img = bg + m * (fill - bg + tex)
    img += rng.normal(0, 0.03, img.shape)
    return np.clip(img, 0, 1)


def vision_train_pool(n, seed, p_congruent=1.0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, N_CLASSES, n)
    X = np.empty((n, S, S), np.float32)
    for i in range(n):
        t = y[i] if rng.random() < p_congruent else rng.integers(0, N_CLASSES)
        X[i] = render(y[i], t, rng)
    return X, y


def vision_test_sets(n, seed):
    rng = np.random.default_rng(seed + 10_000)
    out = {}
    y = rng.integers(0, N_CLASSES, n)
    out["congruent"] = (np.stack([render(c, c, rng) for c in y]), y, None)
    out["shape_only"] = (np.stack([render(c, 0, rng, flat=True) for c in y]), y, None)
    out["texture_only"] = (np.stack([render(None, c, rng, full_texture=True) for c in y]), y, None)
    t = (y + rng.integers(1, N_CLASSES, n)) % N_CLASSES
    out["conflict"] = (np.stack([render(c, tt, rng) for c, tt in zip(y, t)]), y, t)
    return out


# ---------------- audio ----------------
_FORMANTS = [  # (centre bins, as fraction of 32 frequency rows); class-specific
    (0.15, 0.45), (0.15, 0.70), (0.25, 0.55), (0.25, 0.85),
    (0.35, 0.60), (0.35, 0.80), (0.20, 0.35), (0.40, 0.90),
]
_SWEEP = [0, 0, 1, -1, 0, 1, -1, 0]  # formant glide direction over time


def render_audio(cls, rng, noise=0.35):
    f = np.arange(S, dtype=np.float32)[None, :]          # frequency rows
    t = np.arange(S, dtype=np.float32)[:, None] / S      # time
    shift = rng.uniform(-0.05, 0.05)
    spec = np.zeros((S, S), np.float32)
    f0 = rng.uniform(1.6, 2.6)                           # pitch -> harmonic spacing
    harm = 0.5 + 0.5 * np.cos(2 * np.pi * f / f0) ** 8
    for k, fc in enumerate(_FORMANTS[cls]):
        c = (fc + shift + 0.08 * _SWEEP[cls] * (t - 0.5) * (k + 1)) * S
        spec += np.exp(-((f - c) ** 2) / (2 * 2.2 ** 2))
    on, off = rng.uniform(0.05, 0.25), rng.uniform(0.75, 0.95)
    env = ((t > on) & (t < off)).astype(np.float32)
    spec = spec * harm * env
    spec += rng.normal(0, noise, spec.shape)  # noisy so audio alone is imperfect
    return np.clip(0.2 + 0.6 * spec, 0, 1).astype(np.float32)


def av_pool(n, seed, vision_noise=0.0, audio_noise=0.35):
    """Paired image + sound sharing a label. Images use congruent textures."""
    rng = np.random.default_rng(seed + 777)
    y = rng.integers(0, N_CLASSES, n)
    V = np.empty((n, S, S), np.float32)
    A = np.empty((n, S, S), np.float32)
    for i in range(n):
        t = y[i] if rng.random() < P_CONGRUENT else rng.integers(0, N_CLASSES)
        img = render(y[i], t, rng)
        if vision_noise:
            img = np.clip(img + rng.normal(0, vision_noise, img.shape), 0, 1)
        V[i] = img
        A[i] = render_audio(y[i], rng, audio_noise)
    return V, A, y
