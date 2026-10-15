"""E1: how to model age (snapshot / grow / overproduce-prune / hierarchical
critical periods / separate per-age models), with and without the paper's
acuity+contrast schedule. E2: critical period for an early 'cataract' deficit."""
import os
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
import json, sys, time, itertools
import multiprocessing as mp
import numpy as np

SPM = 100          # optimizer steps per simulated month
BATCH = 64
LR = 2e-3
WIDTH = 16
GROW_WIDTHS = {0: 4, 3: 8, 6: 12, 9: 16}   # active channels by month
SEP_WIDTH = {3: 4, 6: 8, 9: 12, 12: 16}
PRUNE_WIDTH, PRUNE_START, PRUNE_END, PRUNE_DENSITY = 24, 8, 12, 0.45
CATARACT_SIGMA = 3.0


def input_at(month, cfg):
    from devlib import age_params
    if cfg.get("deficit", 0) and cfg.get("deficit_start", 0) <= month < cfg.get("deficit_start", 0) + cfg["deficit"]:
        return CATARACT_SIGMA, 1.0
    if cfg["input"] == "dev":
        return age_params(month)
    return 0.0, 1.0


def run(cfg):
    try:
        os.sched_setaffinity(0, {(mp.current_process()._identity[0] - 1) % 4})
    except Exception:
        pass
    import jax, jax.numpy as jnp
    import stimuli as st
    from devlib import (init_net, channel_masks, grow_to, magnitude_masks, ones_masks,
                        active_params, adam_init, train_step, evaluate, kernel)
    spm = cfg.get("spm", SPM)
    seed = cfg["seed"]
    X, y = st.vision_train_pool(cfg.get("pool", 40_000), seed)
    tests = st.vision_test_sets(600, seed)
    tests = {k: (jnp.asarray(a), b, c) for k, (a, b, c) in tests.items()}
    rng = np.random.default_rng(seed + 1)
    key = jax.random.PRNGKey(seed)
    strat = cfg["strategy"]
    months = cfg.get("months", 12)
    if strat == "separate":
        age = cfg["age"]
        width, months_train, fixed_age = SEP_WIDTH[age], age, age
    else:
        width = PRUNE_WIDTH if strat == "prune" else WIDTH
        months_train, fixed_age = months, None
    key, sub = jax.random.split(key)
    p = init_net(sub, width)
    cm = channel_masks(width, GROW_WIDTHS[0] / WIDTH if strat == "grow" else 1.0)
    wm = ones_masks(p)
    s = adam_init(p)
    lr_tree = {k: jnp.asarray(LR) for k in p}
    traj, t0 = [], time.time()
    for step in range(months_train * spm):
        month = step / spm
        if strat == "grow" and step % spm == 0 and int(month) in GROW_WIDTHS and month > 0:
            p, cm = grow_to(p, cm, GROW_WIDTHS[int(month)] / WIDTH)
        if strat == "hiercp" and step % 20 == 0:   # plasticity closes bottom-up
            mult = {}
            for ks, (a, b) in ((("c1", "b1"), (3, 6)), (("c2", "b2"), (6, 9))):
                f = min(1.0, max(0.0, (month - a) / (b - a)))
                for k in ks:
                    mult[k] = 1 - 0.95 * f
            lr_tree = {k: jnp.asarray(LR * mult.get(k, 1.0)) for k in p}
        if strat == "prune" and month >= PRUNE_START and step % 20 == 0:
            f = min(1.0, (month - PRUNE_START) / (PRUNE_END - PRUNE_START - 1))
            dens = PRUNE_DENSITY + (1 - PRUNE_DENSITY) * (1 - f) ** 3   # Zhu & Gupta cubic
            wm = magnitude_masks(p, dens)
        sig, gain = input_at(fixed_age if fixed_age is not None else month, cfg)
        idx = rng.integers(0, X.shape[0], BATCH)
        key, sub = jax.random.split(key)
        p, s, loss = train_step(p, s, cm, wm, lr_tree, jnp.asarray(X[idx]), jnp.asarray(y[idx]),
                                kernel(sig), gain, sub)
        if (step + 1) % spm == 0 and (fixed_age is None or step + 1 == months_train * spm):
            m = (step + 1) // spm
            r = {"month": m if fixed_age is None else fixed_age, "params": active_params(p, cm, wm),
                 "loss": round(float(loss), 4)}
            r.update({k: round(v, 4) for k, v in evaluate(p, cm, tests).items()})
            sa, ga = input_at(m - 1e-6 if fixed_age is None else fixed_age, cfg)
            r["behavioral"] = round(evaluate(p, cm, {"congruent": tests["congruent"]}, sa, ga)["congruent"], 4)
            traj.append(r)
    return {"cfg": cfg, "traj": traj, "secs": round(time.time() - t0, 1)}


def e1_cfgs(seeds):
    out = []
    for seed, inp in itertools.product(seeds, ["dev", "clear"]):
        for s in ["fixed", "grow", "prune", "hiercp"]:
            out.append({"exp": "E1", "strategy": s, "input": inp, "seed": seed})
        for a in [3, 6, 9, 12]:
            out.append({"exp": "E1", "strategy": "separate", "age": a, "input": inp, "seed": seed})
    return out


def e2_cfgs(seeds):
    out = []
    for seed, s, k in itertools.product(seeds, ["fixed", "prune"], [0, 3, 6, 9]):
        out.append({"exp": "E2", "strategy": s, "input": "clear", "deficit": k,
                    "months": k + 8, "seed": seed})
    return out


def e2b_cfgs(seeds):
    # sliding 3-month deprivation window: is sensitivity age-dependent?
    return [{"exp": "E2b", "strategy": "fixed", "input": "clear", "deficit": 3, "deficit_start": st0,
             "months": st0 + 3 + 8, "seed": seed} for seed in seeds for st0 in [0, 3, 6, 9]]


def e1x_cfgs(seeds):
    return [{"exp": "E1x", "strategy": "fixed", "input": inp, "seed": seed}
            for seed in [3, 4, 5, 6] for inp in ["dev", "clear"]]


if __name__ == "__main__":
    which = sys.argv[1]
    seeds = [0, 1, 2]
    cfgs = {"E1": e1_cfgs, "E2": e2_cfgs, "E2b": e2b_cfgs, "E1x": e1x_cfgs}[which](seeds)
    if len(sys.argv) > 2:
        sel = sys.argv[2]
        cfgs = [dict(c, spm=20, pool=3000) for c in cfgs if c["seed"] == 0][: int(sel)]
    cfgs.sort(key=lambda c: -c.get("months", c.get("age", 12)))
    with mp.get_context("spawn").Pool(4) as pool:
        res = []
        for r in pool.imap_unordered(run, cfgs):
            res.append(r)
            print(json.dumps(r["cfg"]), r["secs"], json.dumps(r["traj"][-1]), flush=True)
    tag = "smoke" if len(sys.argv) > 2 else ""
    json.dump(res, open(f"results_{which}{tag}.json", "w"))
