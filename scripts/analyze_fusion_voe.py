"""Violation of expectation for the *fused* world model, from saved grown checkpoints.

For each seed and stage: rebuild the grown sense models, retrain the fusion model
(same recipe as run_experiments), then compare surprise on possible vs impossible
occlusion events. Surprise = error of the fused model's one-step prediction of each
sense latent, normalised per sense, averaged over the hidden stretch and the two
frames after the toy should re-emerge.

Usage: python scripts/analyze_fusion_voe.py [--seeds 0,1,2] [--results results]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dwm import capability as cap  # noqa: E402
from dwm.env import generate  # noqa: E402
from dwm.fusion import SENSES, sense_latents, train_fusion  # noqa: E402
from dwm.models import SenseWorldModel  # noqa: E402
from dwm.train import to_torch  # noqa: E402


def load_stage(results, seed, sense, stage_idx):
    ck = torch.load(os.path.join(results, "grown", f"seed{seed}", f"{sense}_grown_stage{stage_idx}.pt"))
    m = SenseWorldModel(sense, ck["width"], ck["window"])
    m.load_state_dict(ck["state"])
    return m.eval()


@torch.no_grad()
def fused_surprise(fus, zs, action, stats):
    B, T = action.shape[:2]
    mask = {s: torch.ones(B, T, 1) for s in SENSES}
    _, state = fus(zs, mask, action)
    per = {}
    for s in SENSES:
        pred = fus.read[s](state[:, :-1])
        per[s] = (((pred - zs[s][:, 1:]) ** 2).mean(-1) / stats[s]).numpy()
    per["all"] = sum(per[s] for s in SENSES) / len(SENSES)
    return per  # each (B, T-1); index t-1 predicts frame t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--results", default="results")
    ap.add_argument("--steps", type=int, default=3000)
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    out = {}
    for seed in [int(s) for s in args.seeds.split(",")]:
        s = 1000 * seed
        train = to_torch(generate(3000, seed=s + 1))
        pos = to_torch(generate(400, seed=s + 3, voe="possible"))
        imp = to_torch(generate(400, seed=s + 3, voe="impossible"))
        vis = pos["visible"].numpy()
        out[seed] = {}
        for i, age in enumerate(cap.STAGES):
            models = {sn: load_stage(args.results, seed, sn, i) for sn in SENSES}
            k = cap.capacity(age)["window"]
            ztr = sense_latents(models, train, age)
            fus = train_fusion(ztr, train["action"], window=k, steps=args.steps, seed=seed).eval()
            stats = {sn: float(ztr[sn].var()) for sn in SENSES}
            ep = fused_surprise(fus, sense_latents(models, pos, age), pos["action"], stats)
            ei = fused_surprise(fus, sense_latents(models, imp, age), imp["action"], stats)
            res = {}
            for key in ep:
                sp, si = [], []
                for e in range(vis.shape[0]):
                    hidden = np.where(vis[e] == 0)[0]
                    if len(hidden) == 0 or hidden[-1] + 3 >= vis.shape[1]:
                        continue
                    sl = slice(hidden[0] - 1, hidden[-1] + 3)  # whole hidden stretch + re-emergence
                    sp.append(ep[key][e, sl].mean()); si.append(ei[key][e, sl].mean())
                sp, si = float(np.mean(sp)), float(np.mean(si))
                res[key] = {"surprise_possible": sp, "surprise_impossible": si, "voe_index": (si - sp) / (si + sp)}
            out[seed][age] = res
            print(seed, age, {k: round(v['voe_index'], 3) for k, v in out[seed][age].items()}, flush=True)
    with open(os.path.join(args.results, "fusion_voe.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
