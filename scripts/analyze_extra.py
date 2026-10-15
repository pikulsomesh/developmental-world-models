"""Analyses requested in the internal peer review (report/PEER_REVIEW.md).

1. own_age   : probes fitted and tested on input at each checkpoint's *own* age, so
               regimes are not compared only on adult-quality input (distribution shift).
2. ceiling   : how much each sense's front end lets through at each age, measured by a
               small supervised network trained directly on filtered input (an upper
               bound on what any self-supervised model could extract from one frame).
3. stats     : bootstrap 95% CIs, Welch t-tests and Hedges' g for the key comparisons.
4. compute   : training compute per regime (parameter-updates), so growth is compared
               at matched updates *and* reported at its lower compute.
5. milestones: age at which each sense's ceiling first reaches half its 12-month value,
               and Kendall's tau between the model's and infants' order of senses.

Usage: python scripts/analyze_extra.py [--parts own_age,ceiling,stats,compute,milestones]
Writes results/extra.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dwm import capability as cap  # noqa: E402
from dwm import senses as S  # noqa: E402
from dwm.env import generate  # noqa: E402
from dwm.models import SenseWorldModel  # noqa: E402
from dwm.train import probe_model, to_torch  # noqa: E402

REGIMES = ["curriculum_only", "grown", "growth_only", "adult_scratch", "shuffled_order", "reversed_immaturity"]
SCHEDULE = {"curriculum_only": "age", "grown": "age", "growth_only": "adult", "adult_scratch": "adult",
            "shuffled_order": "shuffled", "reversed_immaturity": "reversed"}
CEIL_AGES = [0, 1, 2, 3, 4, 6, 9, 12]


def load(path, sense):
    ck = torch.load(path)
    m = SenseWorldModel(sense, ck["width"], ck["window"])
    m.load_state_dict(ck["state"])
    return m.eval()


def own_age(results, out):
    """Probe each checkpoint on input of the age that checkpoint stands for."""
    res = out.get("own_age", {})  # resumable: finished (regime, seed) cells are kept
    cache = {}
    xpath = os.path.join(results, "extra.json")
    for regime in REGIMES:
        for path in sorted(glob.glob(os.path.join(results, regime, "seed*", "vision_*_stage0.pt"))):
            seed = int(path.split(os.sep)[-2][4:])
            if seed in res.get(regime, {}).get("_seeds", []):
                continue
            if seed not in cache:
                s = 1000 * seed
                cache = {seed: (to_torch(generate(1500, seed=s + 1)), to_torch(generate(400, seed=s + 2)))}
            train, test = cache[seed]
            for sense in ["vision", "audio"]:
                for i, age in enumerate(cap.STAGES):
                    p = os.path.join(results, regime, f"seed{seed}", f"{sense}_{regime}_stage{i}.pt")
                    if not os.path.exists(p):
                        continue
                    m = load(p, sense)
                    r = probe_model(m, train, test, age)
                    res.setdefault(regime, {}).setdefault(sense, {}).setdefault(str(age), []).append(r)
                    print("own_age", regime, seed, sense, age, {k: round(v, 3) for k, v in r.items()}, flush=True)
            res.setdefault(regime, {}).setdefault("_seeds", []).append(seed)
            out["own_age"] = res
            with open(xpath, "w") as f:
                json.dump(out, f, indent=1)
    out["own_age"] = res


class Ceiling(nn.Module):
    def __init__(self, sense):
        super().__init__()
        if sense == "vision":
            self.f = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, 1, 1),
                                   nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
                                   nn.Flatten(), nn.Linear(64 * 16, 128), nn.ReLU())
        else:
            self.f = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.pos, self.typ = nn.Linear(128, 2), nn.Linear(128, 4)

    def forward(self, x):
        h = self.f(x)
        return self.pos(h), self.typ(h)


def ceiling(out, steps=1500):
    """Supervised read-out of toy position and identity from one filtered observation."""
    torch.manual_seed(0)
    tr, te = to_torch(generate(1500, seed=901)), to_torch(generate(300, seed=902))
    res = {}

    def frames(d, sense):
        key = "contact" if sense == "touch" else "present"
        mask = d[key].reshape(-1) > 0
        x = d[sense].reshape(-1, *d[sense].shape[2:])[mask]
        pos = d["obj_pos"].reshape(-1, 2)[mask] / 31.0
        typ = d["obj_type"].view(-1, 1).expand(-1, d["present"].shape[1]).reshape(-1)[mask]
        return x, pos, typ

    for sense in ["vision", "audio", "touch"]:
        xtr, ptr, ttr = frames(tr, sense)
        xte, pte, tte = frames(te, sense)
        for age in CEIL_AGES:
            g = torch.Generator().manual_seed(age)
            m = Ceiling(sense)
            opt = torch.optim.Adam(m.parameters(), 1e-3)
            for step in range(steps):
                idx = torch.randint(0, len(xtr), (128,), generator=g)
                x = S.apply(sense, xtr[idx], age, g)
                pp, pt = m(x)
                loss = F.mse_loss(pp, ptr[idx]) * 20 + F.cross_entropy(pt, ttr[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                pp, pt = [], []
                for j in range(0, len(xte), 512):
                    a, b = m(S.apply(sense, xte[j:j + 512], age, g))
                    pp.append(a); pt.append(b)
                pp, pt = torch.cat(pp), torch.cat(pt)
            r2 = 1 - ((pp - pte) ** 2).sum() / ((pte - pte.mean(0)) ** 2).sum()
            acc = (pt.argmax(1) == tte).float().mean()
            res.setdefault(sense, {})[str(age)] = {"pos_r2": float(r2), "type_acc": float(acc)}
            print("ceiling", sense, age, res[sense][str(age)], flush=True)
    out["ceiling"] = res


def _boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    b = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def _welch(a, b):
    from math import erf, sqrt
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / len(a) + vb / len(b))
    t = (a.mean() - b.mean()) / se
    df = (va / len(a) + vb / len(b)) ** 2 / ((va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1))
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), df))
    except Exception:  # normal approximation
        p = float(2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2)))))
    sp = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    g = (a.mean() - b.mean()) / sp * (1 - 3 / (4 * (len(a) + len(b)) - 9))
    return {"t": float(t), "df": float(df), "p": p, "hedges_g": float(g)}


def stats(results, out):
    def vals(regime, sense, key, sub, stage=-1):
        v = []
        for p in sorted(glob.glob(os.path.join(results, regime, "seed*", "metrics.json"))):
            m = json.load(open(p))
            t = m["senses"].get(sense, {}).get(regime)
            if t:
                v.append(t[stage][key][sub] if sub else t[stage][key])
        return v

    comps = [
        ("vision clean: immature vs adult", ("curriculum_only", "adult_scratch", "vision", "probe_adult", "pos_r2")),
        ("vision clean: grown vs immature", ("grown", "curriculum_only", "vision", "probe_adult", "pos_r2")),
        ("vision identity: grown vs immature", ("grown", "curriculum_only", "vision", "probe_adult", "type_acc")),
        ("vision newborn-input: immature vs adult", ("curriculum_only", "adult_scratch", "vision", "probe_newborn", "pos_r2")),
        ("hearing clean: immature vs adult", ("curriculum_only", "adult_scratch", "audio", "probe_adult", "pos_r2")),
        ("hearing clean: grown vs growth-only", ("grown", "growth_only", "audio", "probe_adult", "pos_r2")),
    ]
    res = {"comparisons": {}, "cells": {}}
    for name, (a, b, sense, key, sub) in comps:
        va, vb = vals(a, sense, key, sub), vals(b, sense, key, sub)
        if len(va) >= 2 and len(vb) >= 2:
            res["comparisons"][name] = {"a": va, "b": vb, **_welch(va, vb)}
    for regime in REGIMES:
        for sense in ["vision", "audio"]:
            for key, sub in [("probe_adult", "pos_r2"), ("probe_newborn", "pos_r2"), ("probe_adult", "type_acc")]:
                v = vals(regime, sense, key, sub)
                if v:
                    res["cells"][f"{regime}/{sense}/{key}.{sub}"] = {
                        "n": len(v), "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                        "ci95": _boot_ci(v) if len(v) > 1 else [float(v[0])] * 2}
    fo = os.path.join(results, "grown")
    occ = {}
    for p in sorted(glob.glob(os.path.join(fo, "seed*", "metrics.json"))):
        f = json.load(open(p))["fusion"]["12"]["occluded_tracking"]
        for k, v in f.items():
            occ.setdefault(k, []).append(v["x_r2"])
    if occ:
        res["comparisons"]["hidden toy at 12 mo: fused vs vision memory"] = {
            "a": occ["fused"], "b": occ["vision_memory"], **_welch(occ["fused"], occ["vision_memory"])}
        d = np.array(occ["fused"]) - np.array(occ["vision_memory"])
        res["comparisons"]["hidden toy at 12 mo: fused vs vision memory"]["paired_diff"] = {
            "mean": float(d.mean()), "min": float(d.min()), "max": float(d.max())}
    out["stats"] = res
    print(json.dumps(res["comparisons"], indent=1)[:3000])


def compute(out):
    """Parameter-updates (params x updates) per regime: a proxy for training compute."""
    res = {}
    steps = {"vision": 2000, "audio": 1500, "touch": 1500, "proprio": 1500}
    for sense in ["vision", "audio", "touch", "proprio"]:
        grown = [SenseWorldModel(sense, cap.capacity(a)["width"], cap.capacity(a)["window"]).n_params()
                 for a in cap.STAGES]
        fixed = grown[-1]
        res[sense] = {"params_by_stage_grown": grown, "params_fixed": fixed,
                      "updates_x_params_grown": float(sum(grown) * steps[sense]),
                      "updates_x_params_fixed": float(4 * fixed * steps[sense]),
                      "ratio_grown_to_fixed": float(sum(grown) / (4 * fixed))}
    out["compute"] = res
    print(res)


def milestones(out):
    """Age at which a sense's front end first passes half of its 12-month ceiling."""
    c = out.get("ceiling")
    if not c:
        return
    metric = {"vision": "pos_r2", "audio": "pos_r2", "touch": "type_acc"}
    ages = {}
    for sense, m in metric.items():
        curve = [(int(a), v[m]) for a, v in c[sense].items()]
        curve.sort()
        lo = curve[0][1] if m == "pos_r2" else 0.25
        final = curve[-1][1]
        thr = lo + 0.5 * (final - lo) if m == "type_acc" else 0.5 * final
        ages[sense] = next((a for a, v in curve if v >= thr), None)
    infant = {"touch": 0, "audio": 1, "vision": 2}  # order of functional maturity (Gottlieb 1971)
    from itertools import combinations
    conc = disc = ties = 0
    for a, b in combinations(infant, 2):
        x = np.sign(infant[a] - infant[b])
        y = np.sign((ages[a] or 0) - (ages[b] or 0))
        if y == 0:
            ties += 1
        elif x == y:
            conc += 1
        else:
            disc += 1
    tau = (conc - disc) / max(1, conc + disc + ties)
    out["milestones"] = {"half_ceiling_age_months": ages, "infant_order_rank": infant, "kendall_tau_b_approx": tau,
                         "pairs": {"concordant": conc, "discordant": disc, "ties": ties}}
    print(out["milestones"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--parts", default="stats,compute,ceiling,milestones,own_age")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("THREADS", "2")))
    path = os.path.join(args.results, "extra.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    for part in args.parts.split(","):
        {"own_age": lambda: own_age(args.results, out), "ceiling": lambda: ceiling(out),
         "stats": lambda: stats(args.results, out), "compute": lambda: compute(out),
         "milestones": lambda: milestones(out)}[part]()
        with open(path, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
