"""E3: sensory onset order. Two sense-specific encoders (vision, audition) and a
fusion read-out. Compare: simultaneous onset, audition-first (human order: hearing
starts prenatally through a low-pass womb), vision-first, and separately trained
sense-specific models fused afterwards."""
import os
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
import json, sys, time, itertools, math
import multiprocessing as mp
import numpy as np

SPM = 100
BATCH = 64
LR = 2e-3
W = 12
BIRTH = 3          # months of 'prenatal' audio-only experience in audio_first
WOMB_CUTOFF = 11   # frequency rows above this are attenuated in utero
VISION_NOISE = 0.3
AUDIO_NOISE = 1.2


def run(cfg):
    try:
        os.sched_setaffinity(0, {(mp.current_process()._identity[0] - 1) % 4})
    except Exception:
        pass
    import jax, jax.numpy as jnp
    import stimuli as st
    from devlib import (init_encoder, encode, channel_masks, adam_init, adam_update,
                        perceive, kernel, age_params, xent)
    spm = cfg.get("spm", SPM)
    seed, sched = cfg["seed"], cfg["schedule"]
    months = cfg.get("months", 12)
    vn, an = cfg.get("vn", VISION_NOISE), cfg.get("an", AUDIO_NOISE)
    V, A, y = st.av_pool(cfg.get("pool", 30_000), seed, vision_noise=vn, audio_noise=an)
    Vt, At, yt = st.av_pool(1000, seed + 999, vision_noise=vn, audio_noise=an)
    Vt, At = jnp.asarray(Vt), jnp.asarray(At)
    rng = np.random.default_rng(seed + 1)
    key = jax.random.PRNGKey(seed)
    kv, ka, kf, key = jax.random.split(key, 4)
    cm = channel_masks(W)
    D = 2 * W
    p = {"v": init_encoder(kv, W), "a": init_encoder(ka, W),
         "fc": jax.random.normal(kf, (2 * D, 8)) * math.sqrt(1 / (2 * D)), "bf": jnp.zeros(8)}
    womb = jnp.asarray((np.arange(32) <= WOMB_CUTOFF).astype(np.float32) * 0.9 + 0.1)

    def feats(p, xv, xa, pv, pa):
        return jnp.concatenate([encode(p["v"], cm, xv) * pv, encode(p["a"], cm, xa) * pa], 1)

    def logits(p, xv, xa, pv, pa):
        return feats(p, xv, xa, pv, pa) @ p["fc"] + p["bf"]

    @jax.jit
    def step(p, s, lr_tree, xv, xa, yb, k, gain, pv, pa, in_womb, key):
        k1, k2 = jax.random.split(key)
        xv = perceive(xv, k, gain, k1)
        xa = jnp.where(in_womb, 0.2 + (xa - 0.2) * womb[None, None, :], xa)
        xa = xa + 0.03 * jax.random.normal(k2, xa.shape)
        loss, g = jax.value_and_grad(lambda q: xent(logits(q, xv, xa, pv, pa), yb))(p)
        p, s = adam_update(p, g, s, lr_tree)
        return p, s, loss

    @jax.jit
    def pred(p, xv, xa, pv, pa):
        return logits(p, xv, xa, pv, pa).argmax(1)

    def onset(month):
        """(vision present, audio present, vision age for acuity, audio in womb)"""
        if sched == "simultaneous":
            return 1.0, 1.0, month, False
        if sched == "audio_first":
            if month < BIRTH:
                return 0.0, 1.0, 0.0, True
            return 1.0, 1.0, month - BIRTH, False
        if sched == "vision_first":
            return 1.0, float(month >= BIRTH), month, False
        if sched == "vision_only":
            return 1.0, 0.0, month, False
        if sched == "audio_only":
            return 0.0, 1.0, 0.0, False
        raise ValueError(sched)

    def evaluate(p):
        out = {}
        for name, pv, pa in (("fused", 1.0, 1.0), ("vision_alone", 1.0, 0.0), ("audio_alone", 0.0, 1.0)):
            out[name] = round(float((np.asarray(pred(p, Vt, At, pv, pa)) == yt).mean()), 4)
        return out

    def train(p, schedule_fn, n_steps, lr_tree, t_offset=0.0, record=True):
        s = adam_init(p)
        traj = []
        nonlocal key
        for i in range(n_steps):
            month = t_offset + i / spm
            pv, pa, vage, womb_on = schedule_fn(month)
            sig, gain = age_params(vage)
            idx = rng.integers(0, len(y), BATCH)
            key, sub = jax.random.split(key)
            p, s, loss = step(p, s, lr_tree, jnp.asarray(V[idx]), jnp.asarray(A[idx]),
                              jnp.asarray(y[idx]), kernel(sig), gain, pv, pa, womb_on, sub)
            if record and (i + 1) % spm == 0:
                traj.append(dict(month=round(month + 1 / spm), **evaluate(p)))
        return p, traj

    def probe(p, which):
        """Linear probe on a frozen encoder: quality of that sense's representation."""
        enc = jax.jit(lambda x: encode(p[which], cm, x))
        Xtr = V[:4000] if which == "v" else A[:4000]
        Ftr = np.asarray(enc(jnp.asarray(Xtr)))
        Fte = np.asarray(enc(Vt if which == "v" else At))
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
        Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd
        Wl, bl = jnp.zeros((Ftr.shape[1], 8)), jnp.zeros(8)
        lossf = lambda wb: xent(jnp.asarray(Ftr) @ wb[0] + wb[1], jnp.asarray(y[:4000])) + 1e-3 * (wb[0] ** 2).sum()
        g = jax.jit(jax.grad(lossf))
        wb = (Wl, bl)
        for _ in range(500):
            gw = g(wb)
            wb = (wb[0] - 0.5 * gw[0], wb[1] - 0.5 * gw[1])
        return round(float(((Fte @ np.asarray(wb[0]) + np.asarray(wb[1])).argmax(1) == yt).mean()), 4)

    t0 = time.time()
    full = {k: jnp.asarray(LR) for k in ("v", "a", "fc", "bf")}
    full = jax.tree_util.tree_map(lambda _: jnp.asarray(LR), p)
    if sched == "separate_then_fuse":
        # each sense gets its own model and head, trained alone for 12 months
        global_sched = sched
        out = {}
        traj = []
        for which, solo in (("v", "vision_only"), ("a", "audio_only")):
            q = dict(p)
            sched_fn = (lambda m: (1.0, 0.0, m, False)) if which == "v" else (lambda m: (0.0, 1.0, 0.0, False))
            q, _ = train(q, sched_fn, months * spm, full, record=False)
            p = dict(p, **{which: q[which]})
        # fuse: freeze encoders, train only the read-out for one month
        lr_ro = jax.tree_util.tree_map(lambda _: jnp.asarray(0.0), p)
        lr_ro["fc"], lr_ro["bf"] = jnp.asarray(LR), jnp.asarray(LR)
        p, _ = train(p, lambda m: (1.0, 1.0, 12.0, False), spm, lr_ro, record=False)
        traj = [dict(month=12, **evaluate(p))]
    else:
        p, traj = train(p, onset, months * spm, full)
    res = {"cfg": cfg, "traj": traj, "final": dict(traj[-1], probe_vision=probe(p, "v"),
                                                     probe_audio=probe(p, "a")),
           "secs": round(time.time() - t0, 1)}
    return res


if __name__ == "__main__":
    seeds = [0, 1, 2]
    scheds = ["simultaneous", "audio_first", "vision_first", "separate_then_fuse", "vision_only", "audio_only"]
    cfgs = [{"exp": "E3", "schedule": s, "seed": sd} for sd, s in itertools.product(seeds, scheds)]
    if os.environ.get("E3B"):
        # matched-vision control: 3 prenatal audio months, then 12 postnatal months of both senses
        cfgs = [{"exp": "E3b", "schedule": "audio_first", "months": 15, "seed": sd} for sd in seeds]
    if len(sys.argv) > 1:
        cfgs = [dict(c, spm=20, pool=3000) for c in cfgs if c["seed"] == 0][: int(sys.argv[1])]
    with mp.get_context("spawn").Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 4) as pool:
        res = []
        for r in pool.imap_unordered(run, cfgs):
            res.append(r)
            print(json.dumps(r["cfg"]), r["secs"], json.dumps(r["final"]), flush=True)
    json.dump(res, open("results_E3" + ("b" if os.environ.get("E3B") else "") + ("smoke" if len(sys.argv) > 1 else "") + ".json", "w"))
