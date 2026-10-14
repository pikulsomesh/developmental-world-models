"""Phase-0 experiments: per-sense developmental world models, growth controls, fusion.

Usage:  python scripts/run_experiments.py [--quick] [--seed 0] [--out results]

Writes results/<seed>/metrics.json and model checkpoints. CPU only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dwm import capability as cap  # noqa: E402
from dwm.env import generate  # noqa: E402
from dwm.fusion import run_fusion  # noqa: E402
from dwm.models import SenseWorldModel, clone  # noqa: E402
from dwm.train import SENSES, evaluate_sense, to_torch, train_sense, voe_index  # noqa: E402

CONDITIONS = {
    # name: (grows?, input schedule over the four checkpoints)
    #   "age"      sensory immaturity follows age 0 -> 3 -> 6 -> 12
    #   "adult"    adult-quality input throughout
    #   "shuffled" every batch draws its input quality from a random stage (same data, no order)
    #   "reversed" sharp first, immature last: 12 -> 6 -> 3 -> 0
    "grown": (True, "age"),
    "adult_scratch": (False, "adult"),
    "curriculum_only": (False, "age"),
    "growth_only": (True, "adult"),
    "shuffled_order": (False, "shuffled"),
    "reversed_immaturity": (False, "reversed"),
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--senses", default=",".join(SENSES))
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    n_train, n_test, n_voe = (400, 100, 100) if args.quick else (3000, 400, 400)
    steps = {"vision": 2000, "audio": 1500, "touch": 1500, "proprio": 1500}
    if args.quick:
        steps = {k: 60 for k in steps}

    out = os.path.join(args.out, f"seed{args.seed}")
    os.makedirs(out, exist_ok=True)
    log("generating data")
    s = 1000 * args.seed
    train = to_torch(generate(n_train, seed=s + 1))
    test = to_torch(generate(n_test, seed=s + 2))
    voe_pos = to_torch(generate(n_voe, seed=s + 3, voe="possible"))
    voe_imp = to_torch(generate(n_voe, seed=s + 3, voe="impossible"))

    metrics = {"stages": cap.STAGES, "steps_per_stage": steps, "senses": {}, "fusion": {}}
    snapshots = {age: {} for age in cap.STAGES}
    for sense in args.senses.split(","):
        metrics["senses"][sense] = {}
        for cond in args.conditions.split(","):
            grows, schedule = CONDITIONS[cond]
            torch.manual_seed(args.seed)
            final = cap.capacity(12)
            m = SenseWorldModel(sense, 1, 1) if grows else SenseWorldModel(sense, final["width"], final["window"])
            traj = []
            t0 = time.time()
            for i, age in enumerate(cap.STAGES):
                if grows and i > 0:
                    c = cap.capacity(age)
                    m.grow(c["width"], c["window"])
                train_age = {"age": age, "adult": 12, "shuffled": list(cap.STAGES),
                             "reversed": cap.STAGES[::-1][i]}[schedule]
                own_age = age if schedule == "shuffled" else train_age
                train_sense(m, train, train_age, steps[sense], seed=args.seed * 100 + i)
                ev = evaluate_sense(m, train, test, own_age=own_age)
                if sense == "vision":
                    ev["voe"] = voe_index(m, voe_pos, voe_imp, own_age)
                ev.update(age=age, params=m.n_params(), width=m.width, window=m.window)
                traj.append(ev)
                if cond == "grown":
                    snapshots[age][sense] = clone(m).eval()
                log(sense, cond, f"age {age}", f"{time.time() - t0:.0f}s", json.dumps(_short(ev)))
            metrics["senses"][sense][cond] = traj
            torch.save(m.state_dict(), os.path.join(out, f"{sense}_{cond}.pt"))
            _dump(metrics, out)

    if all(set(v) == set(SENSES) for v in snapshots.values()):
        log("fusion")
        metrics["fusion"] = run_fusion(snapshots, train, test, voe_pos, seed=args.seed,
                                       steps=200 if args.quick else 3000, log=log)
    _dump(metrics, out)
    log("done")


def _short(ev):
    keep = {}
    for k, v in ev.items():
        if isinstance(v, dict):
            keep[k] = {kk: round(vv, 3) for kk, vv in v.items() if isinstance(vv, float)}
        elif isinstance(v, float):
            keep[k] = round(v, 4)
    return keep


def _dump(metrics, out):
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)


if __name__ == "__main__":
    main()
