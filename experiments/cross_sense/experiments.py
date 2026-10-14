"""Run the cross-sense experiments. Usage: python experiments.py {arch,scale,curriculum,fused,all}"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import torch

from xsense import world
from xsense.models import build
from xsense.run import evaluate, fit_and_eval, train

OUT = os.environ.get("XSENSE_OUT", "results")
STEPS = int(os.environ.get("XSENSE_STEPS", 1500))
SEEDS = (0, 1, 2)
FAMILIES = ("ff", "gru", "tf", "pc", "jepa")
BUDGET = 20000


def _init():
    torch.set_num_threads(1)


def _arch(job):
    sense, fam, age, seed = job
    return fit_and_eval(sense, fam, BUDGET, age, STEPS, seed)


def _scale(job):
    sense, budget, age, seed = job
    return fit_and_eval(sense, "gru", budget, age, STEPS, seed)


CURRICULA = {
    "mature":        lambda n: (lambda s: 12),
    "newborn":       lambda n: (lambda s: 0),
    "developmental": lambda n: (lambda s: (0, 3, 6, 12)[min(3, 4 * s // n)]),
    "reversed":      lambda n: (lambda s: (12, 6, 3, 0)[min(3, 4 * s // n)]),
    "prenatal_dev":  lambda n: (lambda s: (-3, 0, 3, 6, 12)[min(4, 5 * s // n)]),
}


def _curr(job):
    sense, cond, seed = job
    t0 = time.time()
    m = build("gru", world.DIMS[sense], BUDGET, seed=seed)
    train(m, sense, CURRICULA[cond](STEPS), STEPS, seed)
    res = []
    for ea in (12, 0):
        r = evaluate(m, sense, ea)
        r.update(sense=sense, condition=cond, eval_age=ea, seed=seed, secs=round(time.time() - t0, 1))
        res.append(r)
    return res


def _fused(job):
    from xsense.fused import eval_fused, train_fused
    sched, seed = job
    m = train_fused(sched, STEPS * 2, seed)
    r = eval_fused(m)
    r.update(schedule=sched, seed=seed)
    return [r]


def run(name, fn, jobs):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    rows = []
    with Pool(4, initializer=_init) as p:
        for i, res in enumerate(p.imap_unordered(fn, jobs)):
            rows.extend(res)
            print(f"[{name}] {i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    with open(f"{OUT}/{name}.json", "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    which = sys.argv[1]
    S = world.SENSES
    if which in ("arch", "all"):
        run("arch", _arch, [(s, f, a, sd) for s in S for f in FAMILIES for a in (0, 12) for sd in SEEDS])
    if which in ("scale", "all"):
        run("scale", _scale, [(s, b, a, sd) for s in S for b in (1500, 6000, 24000, 96000)
                              for a in (0, 12) for sd in SEEDS[:2]])
    if which in ("curriculum", "all"):
        jobs = [(s, c, sd) for s in S for c in ("mature", "newborn", "developmental", "reversed")
                for sd in SEEDS]
        jobs += [("audio", "prenatal_dev", sd) for sd in SEEDS] + [("touch", "prenatal_dev", sd) for sd in SEEDS]
        run("curriculum", _curr, jobs)
    if which in ("fused", "all"):
        from xsense.fused import SCHEDULES
        run("fused", _fused, [(sc, sd) for sc in SCHEDULES for sd in SEEDS])


def _narrow(job):
    from xsense.narrowing import run as nrun
    sense, sched, fam, seed = job
    return nrun(sense, sched, seed, steps=int(os.environ.get("XSENSE_NSTEPS", 3000)), family=fam)


if __name__ == "__main__" and sys.argv[1] == "narrowing":
    run("narrowing", _narrow, [(s, sc, f, sd) for s in world.SENSES for sc in ("developmental", "mature", "exposed")
                               for f in ("gru", "jepa") for sd in SEEDS])
