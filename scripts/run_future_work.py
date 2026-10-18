"""The three follow-up experiments listed as future work in paper 1, run with the main-study code (dwm/).

matched_<sense>  Experience-matched schedule. One continuous run in which the front-end age rises
                 linearly with training steps, 0 -> 12 months over the same total number of updates
                 as the staged schedule (4 x steps per stage). Checkpoints at 1, 3, 6 and 12 months
                 therefore hold experience proportional to age, unlike the staged schedule, where every
                 stage gets the same number of updates. Own-age probe at each checkpoint.
voe_gauss        Uncertainty-aware forecaster. The main-study vision model with a second decoder that
                 predicts a per-pixel log-variance; the prediction loss is the Gaussian negative
                 log-likelihood. Surprise is the NLL of the next frame (also reported: squared error
                 of the predicted mean).
voe_jepa_vr      Latent prediction (EMA target encoder, stop-gradient, VICReg terms, no decoder), the
                 model of paper 2's follow-up runs. Surprise is the squared error between the predicted
                 latent and the target encoder's latent of the next frame.
                 Both voe_* jobs use the staged 0/3/6/12 schedule of the main study and report, at each
                 stage, the own-age probe and the violation-of-expectation index on the main study's
                 possible/impossible events. voe_jepa_vr also writes paper 2's result format, because its
                 training is identical to `vision_jepa_vr` in papers/baby-senses/experiments/age24_jepa.
spectral         No training. For every saved vision checkpoint (runs/<regime>/seed<k>/vision_<regime>_stage<i>.pt,
                 written by scripts/run_experiments.py --out runs/<regime>), the Jacobian of the encoder
                 (input -> latent) on held-out adult-quality frames, expressed in the eigenbasis of the
                 adult input covariance: effective rank of the frame-averaged Jacobian, and the fraction
                 of its energy in the top 10 and 100 input modes.

Usage: python scripts/run_future_work.py --job matched_vision --seed 0
       python scripts/run_future_work.py --job spectral
Writes results/future_work/<job>_seed<k>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "papers", "baby-senses", "experiments", "age24_jepa"))

from dwm import capability as cap  # noqa: E402
from dwm import senses  # noqa: E402
from dwm.env import generate  # noqa: E402
from dwm.models import SenseWorldModel, _wmse  # noqa: E402
from dwm.train import batches, frame_features, latents, probe_model, to_torch, train_sense  # noqa: E402

OUT = os.path.join(ROOT, "results", "future_work")
STEPS = {"vision": 2000, "audio": 1500}
MATCHED_CHECKPOINTS = [1, 3, 6, 12]


# ---------------------------------------------------------------- experience-matched schedule
def train_matched(m, data, total, seed, checkpoints, on_checkpoint, lr=2e-3, bs=32, sub_T=10):
    """Same loop as dwm.train.train_sense, but the front-end age is 12 * step / total."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    obs, act = data[m.sense], data["action"]
    n, step = obs.shape[0], 0
    due = [(round(total * a / 12), a) for a in checkpoints]
    m.train()
    while step < total:
        for idx in batches(n, bs, gen):
            age = 12.0 * step / total
            t0 = int(torch.randint(0, obs.shape[1] - sub_T + 1, (1,), generator=gen))
            o = senses.apply(m.sense, obs[idx, t0:t0 + sub_T], age, gen)
            loss, _ = m.loss(o, act[idx, t0:t0 + sub_T])
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            while due and step >= due[0][0]:
                on_checkpoint(step, due.pop(0)[1])
                m.train()
            if step >= total:
                break


def run_matched(sense, seed):
    torch.manual_seed(seed)
    s = 1000 * seed  # same data seeds as scripts/run_experiments.py
    train, test = to_torch(generate(3000, seed=s + 1)), to_torch(generate(400, seed=s + 2))
    full = cap.capacity(12)
    m = SenseWorldModel(sense, full["width"], full["window"])
    total = 4 * STEPS[sense]
    res = {"job": f"matched_{sense}", "seed": seed, "total_updates": total, "checkpoints": []}
    t0 = time.time()

    def on_checkpoint(step, age):
        r = {"age": age, "updates": step, "probe_own": probe_model(m, train, test, age)}
        res["checkpoints"].append(r)
        print(time.strftime("%H:%M:%S"), res["job"], seed, age, step, f"{time.time() - t0:.0f}s",
              {k: round(v, 3) for k, v in r["probe_own"].items()}, flush=True)
        dump(res, f"matched_{sense}_seed{seed}.json")

    train_matched(m, train, total, seed * 100, MATCHED_CHECKPOINTS, on_checkpoint)


# ---------------------------------------------------------------- uncertainty-aware forecaster
class GaussianWorldModel(SenseWorldModel):
    """Main-study vision model plus a decoder for the per-pixel log-variance of its next-frame forecast.

    The variance head is trained by the Gaussian negative log-likelihood on detached inputs (predicted
    latent and predicted mean), so the forecaster itself receives exactly the main-study gradients and
    the head learns how uncertain that forecaster is, pixel by pixel."""

    LOGVAR_MIN, LOGVAR_RANGE = -7.0, 8.0  # log-variance in [-7, 1]

    def __init__(self, width, window):
        super().__init__("vision", width, window)
        from dwm.models import LATENT, VisionCodec
        self.var_codec = VisionCodec(width, LATENT["vision"])

    def logvar_seq(self, z):
        B, T = z.shape[:2]
        s = self.var_codec.decode(z.reshape(B * T, -1))  # sigmoid output in (0, 1)
        return (self.LOGVAR_MIN + self.LOGVAR_RANGE * s).view(B, T, *s.shape[1:])

    @staticmethod
    def nll(mean, logvar, target):
        return 0.5 * (logvar + (target - mean) ** 2 / logvar.exp())

    def loss(self, o, a):
        base, parts = super().loss(o, a)
        with torch.no_grad():
            _, zhat = self(o, a)
            mean = self.decode_seq(zhat)
        w = self.salience(o)[:, 1:]
        nll = (self.nll(mean, self.logvar_seq(zhat), o[:, 1:]) * w).mean() / w.mean()
        return base + nll, {**parts, "nll": nll.item()}


# ---------------------------------------------------------------- surprise and the VoE index
@torch.no_grad()
def surprise_per_step(model, data, age, kind, seed=123, bs=100):
    """(N, T-1) surprise at each predicted frame: 'mse' of the predicted mean, Gaussian 'nll',
    or 'latent' error against the target encoder."""
    gen = torch.Generator().manual_seed(seed)
    model.eval()
    obs, act = data["vision"], data["action"]
    out = []
    for i in range(0, obs.shape[0], bs):
        o = senses.apply("vision", obs[i:i + bs], age, gen)
        _, zh = model(o, act[i:i + bs])
        if kind == "latent":
            B, T = o.shape[:2]
            zt = model.target.encode(o.reshape(B * T, *o.shape[2:])).view(B, T, -1)
            e = ((zh - zt[:, 1:]) ** 2).mean(-1)
        else:
            mean = model.decode_seq(zh)
            if kind == "mse":
                e = ((mean - o[:, 1:]) ** 2).flatten(2).mean(-1)
            else:
                e = model.nll(mean, model.logvar_seq(zh), o[:, 1:]).flatten(2).mean(-1)
        out.append(e)
    return torch.cat(out)


def voe_from(ep, ei, possible):
    """Same events and frames as dwm.train.voe_index: re-emergence and the next two frames."""
    vis = possible["visible"].numpy()
    s_pos, s_imp = [], []
    for i in range(vis.shape[0]):
        hidden = np.where(vis[i] == 0)[0]
        if len(hidden) == 0:
            continue
        t = hidden[-1] + 1
        if t >= vis.shape[1]:
            continue
        sl = slice(t - 1, min(t + 2, vis.shape[1] - 1))
        s_pos.append(ep[i, sl].mean().item())
        s_imp.append(ei[i, sl].mean().item())
    sp, si = float(np.mean(s_pos)), float(np.mean(s_imp))
    # NLL can be negative, so the index uses the absolute scale of the two surprises
    return {"surprise_possible": sp, "surprise_impossible": si,
            "voe_index": (si - sp) / (abs(si) + abs(sp) + 1e-12)}


def run_voe(job, seed):
    torch.manual_seed(seed)
    s = 1000 * seed
    train, test = to_torch(generate(3000, seed=s + 1)), to_torch(generate(400, seed=s + 2))
    voe_pos = to_torch(generate(400, seed=s + 3, voe="possible"))
    voe_imp = to_torch(generate(400, seed=s + 3, voe="impossible"))
    full = cap.capacity(12)
    if job == "voe_jepa_vr":
        import run as p2  # papers/baby-senses/experiments/age24_jepa/run.py
        m = p2.JEPAWorldModel("vision", full["width"], full["window"], vicreg=True)
        kinds = ["latent"]
        p2res = {"job": "vision_jepa_vr", "seed": seed, "sense": "vision", "acuity_table": "vep",
                 "objective": "jepa_vr", "steps_per_stage": STEPS["vision"], "stages": []}
    else:
        m = GaussianWorldModel(full["width"], full["window"])
        kinds = ["nll", "mse"]
    res = {"job": job, "seed": seed, "stages": []}
    t0 = time.time()
    for i, age in enumerate(cap.STAGES):
        hist = train_sense(m, train, age, STEPS["vision"], seed=seed * 100 + i, log_every=250)
        ev = {"age": age, "probe_own": probe_model(m, train, test, age), "loss_last": hist[-1] if hist else None}
        for k in kinds:
            ev[f"voe_{k}"] = voe_from(surprise_per_step(m, voe_pos, age, k), surprise_per_step(m, voe_imp, age, k), voe_pos)
        res["stages"].append(ev)
        print(time.strftime("%H:%M:%S"), job, seed, age, f"{time.time() - t0:.0f}s",
              {k: round(v, 3) for k, v in ev["probe_own"].items()},
              {k: round(ev[f"voe_{k}"]["voe_index"], 3) for k in kinds}, flush=True)
        dump(res, f"{job}_seed{seed}.json")
        if job == "voe_jepa_vr":
            p2res["stages"].append({"age": age, "probe_own": ev["probe_own"], "spread": p2.spread(m, test, age),
                                    "loss_last": ev["loss_last"], "acuity_cpd": cap.vision_params(age)["acuity_cpd"]})
            with open(os.path.join(p2.OUT, f"vision_jepa_vr_seed{seed}.json"), "w") as f:
                json.dump(p2res, f, indent=1)


# ---------------------------------------------------------------- spectral account for trained networks
def run_spectral(regimes=("curriculum_only", "adult_scratch"), seeds=range(5), n_frames=400):
    """Jacobians are taken on adult-quality frames and expressed in the eigenbasis of the adult input
    covariance, so every checkpoint is measured against the same input modes: what the encoder has
    become sensitive to, ordered from the coarse, high-variance modes down to fine detail."""
    data = to_torch(generate(200, seed=777))
    mask = data["present"].reshape(-1) > 0
    frames = data["vision"].reshape(-1, *data["vision"].shape[2:])[mask]
    g = torch.Generator().manual_seed(0)
    frames = frames[torch.randperm(len(frames), generator=g)[:n_frames]]
    x = senses.apply("vision", frames, 12, torch.Generator().manual_seed(1))
    X = x.flatten(1).double()
    _, sv, Vt = torch.linalg.svd(X - X.mean(0), full_matrices=False)
    res = {"input": {"var_top10": float((sv[:10] ** 2).sum() / (sv ** 2).sum()),
                     "var_top100": float((sv[:100] ** 2).sum() / (sv ** 2).sum())}}
    for regime in regimes:
        for seed in seeds:
            for age_i, age in enumerate(cap.STAGES):
                path = os.path.join(ROOT, "runs", regime, f"seed{seed}", f"vision_{regime}_stage{age_i}.pt")
                if not os.path.exists(path):
                    continue
                ck = torch.load(path, map_location="cpu")
                m = SenseWorldModel("vision", ck["width"], ck["window"])
                m.load_state_dict(ck["state"])
                m.eval()
                J = torch.func.vmap(torch.func.jacrev(lambda v: m.codec.encode(v[None])[0]))(x).detach().flatten(2).double()
                s = torch.linalg.svdvals(J.mean(0))  # frame-averaged Jacobian (latent x input)
                energy = (torch.einsum("nlp,kp->nlk", J, Vt) ** 2).sum((0, 1))  # per adult input mode
                frac = energy / energy.sum()
                r = {"eff_rank": float(s.sum() ** 2 / (s ** 2).sum()), "top10": float(frac[:10].sum()),
                     "top100": float(frac[:100].sum()), "jac_norm": float(J.pow(2).sum((1, 2)).sqrt().mean())}
                res.setdefault(regime, {}).setdefault(str(seed), {})[str(age)] = r
                print("spectral", regime, seed, age, {k: round(v, 3) for k, v in r.items()}, flush=True)
    dump(res, "spectral.json")


def dump(obj, name):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, choices=["matched_vision", "matched_audio", "voe_gauss", "voe_jepa_vr", "spectral"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--steps", type=int, default=0, help="override steps per stage (smoke test)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.steps:
        STEPS = {k: args.steps for k in STEPS}
    if args.job.startswith("matched_"):
        run_matched(args.job.split("_")[1], args.seed)
    elif args.job.startswith("voe_"):
        run_voe(args.job, args.seed)
    else:
        run_spectral()
