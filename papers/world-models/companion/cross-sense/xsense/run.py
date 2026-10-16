"""Training and the shared, sense-agnostic evaluation battery.

Metrics (identical definition for every sense and model family):
  skill        next-step R^2: 1 - err_model / variance of the next input, scored
               against the noise-free (but age-filtered) input so the noise floor
               of immature senses does not dominate
               (0 = no better than "nothing changes", 1 = perfect)
  pos_r2       linear-probe R^2 for object position from the hidden state
  pos_r2_hid   same, only on timesteps where the object is hidden for this
               sense (object permanence analogue)
  id_acc       linear-probe accuracy for object identity (chance = 0.25)
  voe_teleport paired effect size (Cohen's d_z) of prediction error after a
               spatiotemporal-continuity violation vs the matched control
  voe_identity same for an identity-change violation
"""
from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from . import world
from .age import perceive
from .models import build, per_step_error, step_loss

T = 24
T_V = 12          # violation onset
BURN = 3


def batch(sense, age, B, rng, violation=None, with_clean=False):
    lat = world.sample_latents(B, T, rng, violation=violation, t_v=T_V)
    R = world.render(sense, lat)
    X = torch.from_numpy(perceive(sense, R, age, rng))
    if with_clean:
        return X, lat, torch.from_numpy(perceive(sense, R, age, rng, noise_on=False))
    return X, lat


def train(model, sense, age_schedule, steps, seed, B=64, lr=2e-3, log=False):
    """age_schedule: callable step -> age, so curricula reuse the same loop."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for s in range(steps):
        x, _ = batch(sense, age_schedule(s), B, rng)
        loss = step_loss(model, x)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if hasattr(model, "update_target"):
            model.update_target()
        if log and s % 200 == 0:
            print(s, float(loss), flush=True)
    return model


def _probe(H, lat, mask_hidden, rng):
    N = H.shape[0]
    idx = rng.permutation(N)
    tr, te = idx[: N // 2], idx[N // 2:]
    sl = slice(BURN, None)
    def flat(a, ii):
        return a[ii][:, sl].reshape(-1, *a.shape[2:])
    sc = StandardScaler().fit(flat(H, tr))
    Htr, Hte = sc.transform(flat(H, tr)), sc.transform(flat(H, te))
    xtr, xte = flat(lat["x"], tr), flat(lat["x"], te)
    r = Ridge(alpha=1.0).fit(Htr, xtr)
    px = r.predict(Hte)
    out = {"pos_r2": r2_score(xte, px)}
    m = flat(mask_hidden, te)
    out["pos_r2_hid"] = r2_score(xte[m], px[m]) if m.sum() > 20 else float("nan")
    ktr, kte = flat(lat["k"], tr), flat(lat["k"], te)
    clf = LogisticRegression(max_iter=300).fit(Htr, ktr)
    out["id_acc"] = clf.score(Hte, kte)
    return out


@torch.no_grad()
def evaluate(model, sense, age, seed=12345, N=512):
    model.eval()
    rng = np.random.default_rng(seed)
    x, lat, xc = batch(sense, age, N, rng, with_clean=True)
    err, base, h = per_step_error(model, x, clean=xc)
    out = {"skill": float(1 - err[:, BURN:].mean() / base[:, BURN:].mean())}
    # hidden state at step t has seen x_0..x_t; align latents to the input steps
    lat_in = {k: v[:, :-1] for k, v in lat.items()}
    mask = world.unobserved_mask(sense, lat_in)
    out.update(_probe(h.numpy(), lat_in, mask, np.random.default_rng(seed + 1)))
    for vio in ("teleport", "identity"):
        rc = np.random.default_rng(seed + 7)
        rv = np.random.default_rng(seed + 7)
        xc, _ = batch(sense, age, N, rc)
        xv, _ = batch(sense, age, N, rv, violation=vio)
        ec, _, _ = per_step_error(model, xc)
        ev, _, _ = per_step_error(model, xv)
        # error on predicting steps T_V .. T_V+3 (prediction index t predicts x_{t+1})
        w = slice(T_V - 1, T_V + 3)
        d = (ev[:, w].mean(1) - ec[:, w].mean(1)).numpy()
        out[f"voe_{vio}"] = float(d.mean() / (d.std() + 1e-8))
    model.train()
    return out


def fit_and_eval(sense, family, budget, age, steps, seed, eval_ages=None):
    t0 = time.time()
    m = build(family, world.DIMS[sense], budget, seed=seed)
    train(m, sense, lambda s: age, steps, seed)
    res = []
    for ea in (eval_ages or [age]):
        r = evaluate(m, sense, ea)
        r.update(sense=sense, family=family, budget=budget, core_params=m.core_params(),
                 train_age=age, eval_age=ea, seed=seed, secs=round(time.time() - t0, 1))
        res.append(r)
    return res
