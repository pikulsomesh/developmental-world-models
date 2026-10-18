"""Two follow-up experiments for paper 2, built on the main-study code (dwm/), which is not edited.

A. Training to 24 months. The main-study schedule (curriculum_only: full-size network, input
   quality follows age) is extended from 0/3/6/12 to 0/3/6/12/18/24 months.
     vision_vep  : evoked-potential acuity table used by the models (flat after 12 months)
     vision_look : behavioural acuity-card table of the website (docs/app/lab-data.js, ACUITY.look),
                   which keeps rising to 10.5 cycles/degree at 24 months
     audio       : hearing tables (flat after 12 months)
B. JEPA-style latent prediction: an EMA target encoder with stop-gradient and no pixel decoder, on
   the 0/3/6/12 vision schedule. vision_jepa is the plain objective; vision_jepa_vr adds VICReg
   variance and covariance terms on the online latents (Bardes et al. 2022), because the plain
   objective collapsed in a smoke test. The pixel-prediction baseline is vision_vep at 0-12.

Every stage reports the own-age linear probe (position R^2, identity accuracy) and the spread of the
latent code (collapse check). `--ceiling` runs the supervised information ceiling of
scripts/analyze_extra.py for each front end and stage.

Usage: python run.py --job vision_vep --seed 0      (one job per process)
       python run.py --ceiling vision_look
Writes results/<job>_seed<k>.json and results/ceiling_<front end>.json next to this file.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from dwm import capability as cap  # noqa: E402
from dwm import senses as S  # noqa: E402
from dwm.env import generate  # noqa: E402
from dwm.models import SenseWorldModel, windows  # noqa: E402
from dwm.train import frame_features, latents, probe_model, to_torch, train_sense  # noqa: E402

OUT = os.path.join(HERE, "results")
STAGES_24 = [0, 3, 6, 12, 18, 24]
# Behavioural (acuity-card) table of the Baby Senses Lab, docs/app/lab-data.js, ACUITY.look.
ACUITY_LOOK = {0: 0.8, 1: 1.0, 2: 1.8, 3: 2.6, 4: 3.4, 6: 4.8, 9: 6.0, 12: 7.5, 18: 9.0, 24: 10.5, 60: 30}
ACUITY_VEP = dict(cap.VISION_ACUITY_CPD)
JOBS = {  # sense, stages, acuity table, objective
    "vision_vep": ("vision", STAGES_24, "vep", "pixel"),
    "vision_look": ("vision", STAGES_24, "look", "pixel"),
    "audio": ("audio", STAGES_24, "vep", "pixel"),
    "vision_jepa": ("vision", cap.STAGES, "vep", "jepa"),
    "vision_jepa_vr": ("vision", cap.STAGES, "vep", "jepa_vr"),
}
STEPS = {"vision": 2000, "audio": 1500}


def set_acuity(table):
    cap.VISION_ACUITY_CPD = ACUITY_LOOK if table == "look" else ACUITY_VEP


class JEPAWorldModel(SenseWorldModel):
    """Latent prediction: the online encoder and dynamics predict the EMA target encoder's latent of
    the next frame (stop-gradient). The pixel decoder is never used, so it receives no gradient."""

    def __init__(self, sense, width, window, ema=0.99, vicreg=False):
        super().__init__(sense, width, window)
        self.ema, self.vicreg = ema, vicreg
        object.__setattr__(self, "target", copy.deepcopy(self.codec).requires_grad_(False))

    @torch.no_grad()
    def _ema_update(self):
        for pt, po in zip(self.target.parameters(), self.codec.parameters()):
            pt.mul_(self.ema).add_(po.detach(), alpha=1 - self.ema)

    def loss(self, o, a):
        if self.training:
            self._ema_update()  # update from the previous optimiser step (one-step lag)
        z, zhat = self(o, a)
        with torch.no_grad():
            B, T = o.shape[:2]
            zt = self.target.encode(o.reshape(B * T, *o.shape[2:])).view(B, T, -1)
        pred = F.mse_loss(zhat, zt[:, 1:])
        if not self.vicreg:
            return pred, {"pred": pred.item()}
        zz = z.reshape(-1, z.shape[-1])
        zc = zz - zz.mean(0)
        c = zc.T @ zc / (len(zz) - 1)
        var = F.relu(1 - zz.std(0)).mean()
        cov = (c - torch.diag(torch.diag(c))).pow(2).sum() / zz.shape[1]
        return 25 * pred + 25 * var + cov, {"pred": pred.item(), "var": var.item(), "cov": cov.item()}


@torch.no_grad()
def spread(model, test, age):
    """Collapse check on test latents: mean per-dimension s.d. and participation-ratio dimension."""
    z, _ = latents(model, test, age)
    Z = frame_features(z)
    Z = Z - Z.mean(0)
    ev = np.clip(np.linalg.eigvalsh(np.cov(Z.T)), 0, None)
    eff = float(ev.sum() ** 2 / (ev ** 2).sum()) if ev.sum() > 0 else 0.0
    return {"z_std": float(Z.std(0).mean()), "eff_dim": eff,
            "z_dims": int(Z.shape[1])}


def run_job(job, seed):
    sense, stages, table, objective = JOBS[job]
    set_acuity(table)
    torch.manual_seed(seed)
    s = 1000 * seed  # same data seeds as scripts/run_experiments.py
    train, test = to_torch(generate(3000, seed=s + 1)), to_torch(generate(400, seed=s + 2))
    full = cap.capacity(12)
    if objective == "pixel":
        m = SenseWorldModel(sense, full["width"], full["window"])
    else:
        m = JEPAWorldModel(sense, full["width"], full["window"], vicreg=objective == "jepa_vr")
    res = {"job": job, "seed": seed, "sense": sense, "acuity_table": table, "objective": objective,
           "steps_per_stage": STEPS[sense], "stages": []}
    t0 = time.time()
    for i, age in enumerate(stages):
        hist = train_sense(m, train, age, STEPS[sense], seed=seed * 100 + i, log_every=250)
        ev = {"age": age, "probe_own": probe_model(m, train, test, age), "spread": spread(m, test, age),
              "loss_last": hist[-1] if hist else None, "acuity_cpd": cap.vision_params(age)["acuity_cpd"]}
        res["stages"].append(ev)
        print(time.strftime("%H:%M:%S"), job, seed, age, f"{time.time() - t0:.0f}s",
              {k: round(v, 3) for k, v in ev["probe_own"].items()}, ev["spread"], flush=True)
        with open(os.path.join(OUT, f"{job}_seed{seed}.json"), "w") as f:
            json.dump(res, f, indent=1)


def run_ceiling(front):
    from analyze_extra import Ceiling  # supervised read-out network of the main study
    sense = "audio" if front == "audio" else "vision"
    set_acuity("look" if front == "vision_look" else "vep")
    torch.manual_seed(0)
    tr, te = to_torch(generate(1500, seed=901)), to_torch(generate(300, seed=902))

    def frames(d):
        mask = d["present"].reshape(-1) > 0
        x = d[sense].reshape(-1, *d[sense].shape[2:])[mask]
        pos = d["obj_pos"].reshape(-1, 2)[mask] / 31.0
        typ = d["obj_type"].view(-1, 1).expand(-1, d["present"].shape[1]).reshape(-1)[mask]
        return x, pos, typ

    (xtr, ptr, ttr), (xte, pte, tte) = frames(tr), frames(te)
    res = {}
    for age in STAGES_24:
        g = torch.Generator().manual_seed(age)
        m = Ceiling(sense)
        opt = torch.optim.Adam(m.parameters(), 1e-3)
        for _ in range(1500):
            idx = torch.randint(0, len(xtr), (128,), generator=g)
            pp, pt = m(S.apply(sense, xtr[idx], age, g))
            loss = F.mse_loss(pp, ptr[idx]) * 20 + F.cross_entropy(pt, ttr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            outs = [m(S.apply(sense, xte[j:j + 512], age, g)) for j in range(0, len(xte), 512)]
            pp, pt = torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])
        r2 = 1 - ((pp - pte) ** 2).sum() / ((pte - pte.mean(0)) ** 2).sum()
        res[str(age)] = {"pos_r2": float(r2), "type_acc": float((pt.argmax(1) == tte).float().mean())}
        print("ceiling", front, age, res[str(age)], flush=True)
        with open(os.path.join(OUT, f"ceiling_{front}.json"), "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", choices=list(JOBS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ceiling", choices=["vision_vep", "vision_look", "audio"])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--steps", type=int, default=0, help="override steps per stage (smoke test)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(OUT, exist_ok=True)
    if args.steps:
        STEPS = {k: args.steps for k in STEPS}
    if args.ceiling:
        run_ceiling(args.ceiling)
    else:
        run_job(args.job, args.seed)
