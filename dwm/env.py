"""A tiny synthetic multisensory "nursery".

One toy moves around a 32x32 scene and bounces off the walls. The infant's own hand
babbles around the lower half of the view. Each toy type is redundantly specified
across senses (Bahrick & Lickliter's intersensory redundancy):

  vision  : 32x32 RGB frame (toy shape + colour, hand, optional occluder)
  audio   : 2 ears x 32 frequency bins; the toy hums at a type-specific pitch,
            louder when near, panned by horizontal position (sound passes occluders)
  touch   : 8x8 tactile patch centred on the hand (toy shape when in contact)
  proprio : hand position and velocity (4 numbers)
  action  : 2-d motor command that moved the hand

Ground truth (toy position, type, visibility, contact) is kept for probing.
"""
from __future__ import annotations

import numpy as np

H = W = 32
T_DEFAULT = 20
N_TYPES = 4
AUDIO_BINS = 32
TOUCH = 8

_SHAPES = np.array(
    [
        [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],  # square
        [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]],  # plus
        [[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]],  # diamond
        [[1, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 0, 0, 0, 1]],  # x
    ],
    dtype=np.float32,
)
_COLORS = np.array([[0.9, 0.15, 0.1], [0.1, 0.75, 0.2], [0.15, 0.3, 0.95], [0.95, 0.85, 0.1]], np.float32)
_PITCH = np.array([4, 9, 14, 19])
_HAND_RGB = np.array([0.95, 0.75, 0.6], np.float32)
_OCC_RGB = np.array([0.5, 0.5, 0.5], np.float32)
_BG = 0.08


def _stamp(canvas, mask, cx, cy, rgb):
    """Draw a 5x5 mask centred at (cx, cy) with clipping."""
    x0, y0 = int(round(cx)) - 2, int(round(cy)) - 2
    xs0, ys0 = max(0, -x0), max(0, -y0)
    xs1, ys1 = min(5, W - x0), min(5, H - y0)
    if xs0 >= xs1 or ys0 >= ys1:
        return
    m = mask[ys0:ys1, xs0:xs1, None]
    region = canvas[y0 + ys0 : y0 + ys1, x0 + xs0 : x0 + xs1]
    region[:] = region * (1 - m) + rgb * m


def _occupancy(t_type, ox, oy):
    occ = np.zeros((H + 8, W + 8), np.float32)  # padded by 4
    x0, y0 = int(round(ox)) - 2 + 4, int(round(oy)) - 2 + 4
    x0, y0 = min(max(x0, 0), W + 3), min(max(y0, 0), H + 3)
    occ[y0 : y0 + 5, x0 : x0 + 5] = _SHAPES[t_type]
    return occ


def _audio(t_type, ox, oy, rng):
    bins = np.arange(AUDIO_BINS)
    p = _PITCH[t_type]
    spec = np.exp(-0.5 * ((bins - p) / 0.7) ** 2) + 0.5 * np.exp(-0.5 * ((bins - min(2 * p, 31)) / 0.7) ** 2)
    dist = np.hypot(ox - 16, oy - 16)
    amp = 1.0 / (1.0 + dist / 10.0)
    pan = ox / (W - 1)  # 0 = left, 1 = right
    left = spec * amp * (1.2 - 0.8 * pan)
    right = spec * amp * (0.4 + 0.8 * pan)
    out = np.stack([left, right]).astype(np.float32)
    out += 0.01 * rng.standard_normal(out.shape).astype(np.float32)
    return out


def generate(n, T=T_DEFAULT, seed=0, occluder_prob=0.5, voe=None):
    """Generate n episodes.

    voe: None for ordinary episodes, or "possible" / "impossible" for the
    violation-of-expectation probe: the toy crosses a central occluder
    horizontally; in "impossible" it vanishes while hidden (from every sense).
    "track" is like "possible" with a wider occluder and varied speed, used to ask
    where a model thinks the hidden toy is.
    """
    rng = np.random.default_rng(seed)
    vis = np.full((n, T, H, W, 3), _BG, np.float32)
    aud = np.zeros((n, T, 2, AUDIO_BINS), np.float32)
    tch = np.zeros((n, T, TOUCH, TOUCH), np.float32)
    pro = np.zeros((n, T, 4), np.float32)
    act = np.zeros((n, T, 2), np.float32)
    obj = np.zeros((n, T, 2), np.float32)
    typ = np.zeros((n,), np.int64)
    visible = np.zeros((n, T), np.float32)
    present = np.ones((n, T), np.float32)
    contact = np.zeros((n, T), np.float32)

    for i in range(n):
        k = rng.integers(N_TYPES)
        typ[i] = k
        if voe is None:
            ox, oy = rng.uniform(4, 27, size=2)
            ang = rng.uniform(0, 2 * np.pi)
            speed = rng.uniform(0.6, 1.6)
            vx, vy = speed * np.cos(ang), speed * np.sin(ang)
            occ_x = rng.uniform(8, 24) if rng.random() < occluder_prob else None
            occ_w = 8
        elif voe == "track":
            # long occlusion, varied speed and height: where is the toy while hidden?
            ox, oy = 3.0, rng.uniform(5, 14)
            vx, vy = rng.uniform(1.0, 1.8), 0.0
            occ_x, occ_w = 16.0, 14
        else:
            ox, oy = 3.0, rng.uniform(7, 12)
            vx, vy = rng.uniform(1.4, 1.8), 0.0
            occ_x, occ_w = 16.0, 10
        hx, hy = rng.uniform(6, 26), rng.uniform(20, 28)
        a = np.zeros(2)
        alive = True
        for t in range(T):
            # motor babbling: smooth random commands
            a = 0.7 * a + 0.3 * rng.uniform(-1, 1, size=2)
            nhx, nhy = np.clip(hx + 1.5 * a[0], 2, 29), np.clip(hy + 1.5 * a[1], 16, 29)
            hvx, hvy = nhx - hx, nhy - hy
            hx, hy = nhx, nhy
            act[i, t] = a

            hidden = occ_x is not None and abs(ox - occ_x) <= (occ_w / 2 - 2.5)
            if voe == "impossible" and hidden:
                alive = False

            frame = vis[i, t]
            if alive:
                _stamp(frame, _SHAPES[k], ox, oy, _COLORS[k])
            if occ_x is not None:
                x0, x1 = int(round(occ_x - occ_w / 2)), int(round(occ_x + occ_w / 2))
                frame[:, max(0, x0) : min(W, x1)] = _OCC_RGB
            _stamp(frame, _hand_mask(), hx, hy, _HAND_RGB)

            if alive:
                aud[i, t] = _audio(k, ox, oy, rng)
                occ = _occupancy(k, ox, oy)
                cx, cy = int(round(hx)) + 4, int(round(hy)) + 4
                tch[i, t] = occ[cy - 4 : cy + 4, cx - 4 : cx + 4]
            else:
                aud[i, t] = 0.01 * rng.standard_normal((2, AUDIO_BINS))
            pro[i, t] = [hx / 31.0, hy / 31.0, hvx / 1.5, hvy / 1.5]
            obj[i, t] = [ox, oy]
            visible[i, t] = float(alive and not hidden)
            present[i, t] = float(alive)
            contact[i, t] = float(tch[i, t].sum() > 0)

            ox, oy = ox + vx, oy + vy
            if voe is None:
                if ox < 2 or ox > 29:
                    vx, ox = -vx, np.clip(ox, 2, 29)
                if oy < 2 or oy > 29:
                    vy, oy = -vy, np.clip(oy, 2, 29)

    return {
        "vision": vis.transpose(0, 1, 4, 2, 3).copy(),  # (n,T,3,H,W)
        "audio": aud.reshape(n, T, -1),
        "touch": tch.reshape(n, T, -1),
        "proprio": pro,
        "action": act,
        "obj_pos": obj,
        "obj_type": typ,
        "visible": visible,
        "present": present,
        "contact": contact,
    }


def _hand_mask():
    return np.array(
        [[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]], np.float32
    )
