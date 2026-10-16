"""Perceptual narrowing as a cross-sense test.

Infants discriminate non-native phoneme contrasts at 6-8 months and lose that
ability by 10-12 months while native contrasts sharpen (Werker & Tees 1984);
the same happens for other-race / other-species faces (Pascalis et al. 2002;
Kelly et al. 2007) and for audiovisual monkey face-voice matching (Lewkowicz &
Ghazanfar 2006). It is a strong test because it is non-monotonic: a model whose
abilities only ever improve with training cannot show it.

Here the "native" environment only ever contains identities {0, 1}; identities
{2, 3} are the "non-native" contrast and never appear in training. At
checkpoints spread over one continuous run (checkpoint = age), we measure for
both contrasts:
  dprime   separation of the two identities in the hidden state, without
           training a read-out: |mean_a - mean_b| projected on the mean
           difference, over pooled within-class sd (a habituation analogue)
  swap_voe paired d_z of prediction error when the object switches identity
           within the contrast vs a matched no-switch control (dishabituation)
"""
from __future__ import annotations

import numpy as np
import torch

from . import world
from .age import perceive
from .models import build, per_step_error, step_loss
from .run import BURN, T, T_V

NATIVE, NONNATIVE = (0, 1), (2, 3)


def age_at(frac, schedule):
    if schedule == "mature":   # adult front end throughout, isolates experience from sensory maturation
        return 12
    ages = (0, 3, 6, 12)                      # developmental front end over the run
    return ages[min(3, int(frac * 4))]


def _batch(sense, age, B, rng, ids, violation=None):
    lat = world.sample_latents(B, T, rng, violation=violation, t_v=T_V, ids=ids)
    return torch.from_numpy(perceive(sense, world.render(sense, lat), age, rng)), lat


@torch.no_grad()
def measure(model, sense, age, ids, seed=4242, N=384):
    model.eval()
    x, lat = _batch(sense, age, N, np.random.default_rng(seed), ids)
    _, _, h = per_step_error(model, x)
    H = h[:, BURN:].reshape(-1, h.shape[-1]).numpy()
    k = lat["k"][:, :-1][:, BURN:].reshape(-1)
    a, b = H[k == ids[0]], H[k == ids[1]]
    w = a.mean(0) - b.mean(0)
    w = w / (np.linalg.norm(w) + 1e-8)
    pa, pb = a @ w, b @ w
    # split-half: direction from one half of sequences, evaluated on the other, to avoid optimism
    half = H.shape[0] // 2
    a1, b1 = H[:half][k[:half] == ids[0]], H[:half][k[:half] == ids[1]]
    a2, b2 = H[half:][k[half:] == ids[0]], H[half:][k[half:] == ids[1]]
    w = a1.mean(0) - b1.mean(0); w /= (np.linalg.norm(w) + 1e-8)
    pa, pb = a2 @ w, b2 @ w
    dprime = float((pa.mean() - pb.mean()) / np.sqrt(0.5 * (pa.var() + pb.var()) + 1e-8))
    xc, _ = _batch(sense, age, N, np.random.default_rng(seed + 1), ids)
    xv, _ = _batch(sense, age, N, np.random.default_rng(seed + 1), ids, violation="identity")
    ec, _, _ = per_step_error(model, xc)
    ev, _, _ = per_step_error(model, xv)
    wdw = slice(T_V - 1, T_V + 3)
    d = (ev[:, wdw].mean(1) - ec[:, wdw].mean(1)).numpy()
    model.train()
    return {"dprime": dprime, "swap_voe": float(d.mean() / (d.std() + 1e-8))}


def run(sense, schedule, seed, steps=3000, n_ckpt=12, family="gru", budget=20000):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = build(family, world.DIMS[sense], budget, seed=seed)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    ck = set(np.unique(np.round(np.geomspace(1, steps, n_ckpt)).astype(int)) - 1) | {0}
    rows = []
    for s in range(steps + 1):
        frac = s / steps
        age = age_at(frac, schedule)
        if s in ck or s == steps:
            for name, ids in (("native", NATIVE), ("nonnative", NONNATIVE)):
                r = measure(m, sense, age, ids)
                r.update(sense=sense, schedule=schedule, seed=seed, step=s, age=age, contrast=name, family=family)
                rows.append(r)
        if s == steps:
            break
        # "exposed" is the control: same front end, but all four identities occur
        x, _ = _batch(sense, age, 64, rng, None if schedule == "exposed" else NATIVE)
        loss = step_loss(m, x)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if hasattr(m, "update_target"):
            m.update_target()
    return rows
