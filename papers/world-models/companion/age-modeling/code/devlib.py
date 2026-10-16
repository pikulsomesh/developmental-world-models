"""Shared pieces (JAX): small CNN with channel masks (growth) and weight masks
(pruning), developmental input transforms, evaluation."""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import math
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

# Acuity (cpd) and contrast sensitivity tables from the source paper, sec 4.3.1
ACUITY = {0: 1.0, 1: 1.5, 2: 2.0, 3: 4.0, 4: 6.0, 5: 10.0, 6: 15.0, 9: 25.0, 12: 30.0}
CONTRAST = {0: 2, 1: 3, 2: 5, 3: 10, 4: 20, 5: 30, 6: 50, 9: 80, 12: 100}
R = 10  # max blur radius (taps = 2R+1)


def interp(table, a):
    ks = sorted(table)
    if a <= ks[0]:
        return table[ks[0]]
    if a >= ks[-1]:
        return table[ks[-1]]
    for lo, hi in zip(ks[:-1], ks[1:]):
        if lo <= a <= hi:
            t = (a - lo) / (hi - lo)
            return table[lo] * (1 - t) + table[hi] * t


def age_params(age_months):
    """Blur sigma (pixels at 32x32) and contrast gain for an infant of this age."""
    cpd = interp(ACUITY, age_months)
    sigma = 0.15 * 30.0 / cpd
    gain = math.sqrt(interp(CONTRAST, age_months) / 100.0)
    return sigma, gain


def kernel(sigma):
    k = np.zeros(2 * R + 1, np.float32)
    if sigma < 0.3:
        k[R] = 1
    else:
        x = np.arange(-R, R + 1)
        k = np.exp(-x ** 2 / (2 * sigma ** 2)).astype(np.float32)
        k /= k.sum()
    return jnp.asarray(k)


def _conv1d_sep(x, k):
    # x: (B,H,W)
    xp = jnp.pad(x, ((0, 0), (0, 0), (R, R)), mode="reflect")
    x = sum(k[i] * xp[:, :, i:i + x.shape[2]] for i in range(2 * R + 1))
    xp = jnp.pad(x, ((0, 0), (R, R), (0, 0)), mode="reflect")
    return sum(k[i] * xp[:, i:i + x.shape[1], :] for i in range(2 * R + 1))


def perceive(x, k, gain, key, noise=0.03):
    """Optics + retina: blur, contrast loss around image mean, sensor noise. x: (B,H,W)."""
    x = _conv1d_sep(x, k)
    m = x.mean(axis=(1, 2), keepdims=True)
    x = m + (x - m) * gain
    return x + noise * jax.random.normal(key, x.shape)


# ---------------- model ----------------
def init_encoder(key, c):
    k1, k2, k3 = jax.random.split(key, 3)
    he = lambda k, shape: jax.random.normal(k, shape) * math.sqrt(2.0 / (shape[0] * shape[1] * shape[2]))
    return {"c1": he(k1, (3, 3, 1, c)), "b1": jnp.zeros(c),
            "c2": he(k2, (3, 3, c, 2 * c)), "b2": jnp.zeros(2 * c),
            "c3": he(k3, (3, 3, 2 * c, 2 * c)), "b3": jnp.zeros(2 * c)}


def init_net(key, c, n_out=8):
    ke, kf = jax.random.split(key)
    p = init_encoder(ke, c)
    p["fc"] = jax.random.normal(kf, (2 * c, n_out)) * math.sqrt(1.0 / (2 * c))
    p["bf"] = jnp.zeros(n_out)
    return p


def channel_masks(c, frac=1.0):
    def m(n):
        a = np.zeros(n, np.float32)
        a[:max(1, int(round(frac * n)))] = 1
        return jnp.asarray(a)
    return {"m1": m(c), "m2": m(2 * c), "m3": m(2 * c)}


def _conv(x, w, b):
    return jax.lax.conv_general_dilated(x, w, (1, 1), "SAME",
                                        dimension_numbers=("NHWC", "HWIO", "NHWC")) + b


def _pool(x):
    return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")


def encode(p, cm, x):
    x = x[..., None]
    x = _pool(jax.nn.relu(_conv(x, p["c1"], p["b1"])) * cm["m1"])
    x = _pool(jax.nn.relu(_conv(x, p["c2"], p["b2"])) * cm["m2"])
    x = jax.nn.relu(_conv(x, p["c3"], p["b3"])) * cm["m3"]
    return x.mean(axis=(1, 2))


def forward(p, cm, x):
    return encode(p, cm, x) @ p["fc"] + p["bf"]


WEIGHT_KEYS = ["c1", "c2", "c3", "fc"]


def grow_to(p, cm_old, frac):
    """Activate the first `frac` channels; new channels get zero outgoing weights
    so the network's function is preserved at the moment of growth."""
    c = cm_old["m1"].shape[0]
    cm = channel_masks(c, frac)
    p = dict(p)
    for mk, nxt in (("m1", "c2"), ("m2", "c3"), ("m3", "fc")):
        new = (cm[mk] > 0) & (cm_old[mk] == 0)
        if nxt == "fc":
            p[nxt] = jnp.where(new[:, None], 0.0, p[nxt])
        else:
            p[nxt] = jnp.where(new[None, None, :, None], 0.0, p[nxt])
    return p, cm


def magnitude_masks(p, density):
    out = {}
    for k in WEIGHT_KEYS:
        w = np.abs(np.asarray(p[k])).ravel()
        n_keep = max(1, int(density * w.size))
        thr = np.partition(w, w.size - n_keep)[w.size - n_keep]
        out[k] = jnp.asarray((np.abs(np.asarray(p[k])) >= thr).astype(np.float32))
    return out


def ones_masks(p):
    return {k: jnp.ones_like(p[k]) for k in WEIGHT_KEYS}


def active_params(p, cm, wm):
    """Count of non-zero (live) synapses in conv/fc weights."""
    c1, c2, c3 = cm["m1"], cm["m2"], cm["m3"]
    live = {"c1": wm["c1"] * c1, "c2": wm["c2"] * c1[:, None] * c2,
            "c3": wm["c3"] * c2[:, None] * c3, "fc": wm["fc"] * c3[:, None]}
    return int(sum(float(v.sum()) for v in live.values()))


# ---------------- Adam with per-layer lr multipliers ----------------
def adam_init(p):
    z = jax.tree_util.tree_map(jnp.zeros_like, p)
    return {"m": z, "v": jax.tree_util.tree_map(jnp.zeros_like, p), "t": jnp.zeros(())}


def adam_update(p, g, s, lr_tree, b1=0.9, b2=0.999, eps=1e-8):
    t = s["t"] + 1
    m = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, s["m"], g)
    v = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * b * b, s["v"], g)
    mh = jax.tree_util.tree_map(lambda a: a / (1 - b1 ** t), m)
    vh = jax.tree_util.tree_map(lambda a: a / (1 - b2 ** t), v)
    p = jax.tree_util.tree_map(lambda w, a, b, lr: w - lr * a / (jnp.sqrt(b) + eps), p, mh, vh, lr_tree)
    return p, {"m": m, "v": v, "t": t}


def xent(logits, y):
    return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits), y[:, None], 1))


@jax.jit
def train_step(p, s, cm, wm, lr_tree, x, y, k, gain, key):
    x = perceive(x, k, gain, key)
    loss, g = jax.value_and_grad(lambda q: xent(forward(q, cm, x), y))(p)
    p, s = adam_update(p, g, s, lr_tree)
    p = {n: (w * wm[n] if n in wm else w) for n, w in p.items()}
    return p, s, loss


@jax.jit
def predict(p, cm, x, k, gain, key):
    return forward(p, cm, perceive(x, k, gain, key)).argmax(1)


def evaluate(p, cm, tests, sigma=0.0, gain=1.0):
    key = jax.random.PRNGKey(0)
    res = {}
    for name, (X, y, t) in tests.items():
        pr = np.asarray(predict(p, cm, X, kernel(sigma), gain, key))
        if t is None:
            res[name] = float((pr == y).mean())
        else:
            s, tx = (pr == y).sum(), (pr == t).sum()
            res["shape_bias"] = float(s / max(1, s + tx))
            res["conflict_shape_acc"] = float((pr == y).mean())
    if "congruent" in tests:
        X, y, _ = tests["congruent"]
        pr = np.asarray(predict(p, cm, X, kernel(1.5), 1.0, key))
        res["blur_robust"] = float((pr == y).mean())
    return res
