"""Multisensory fusion and the order in which senses come online.

Tests the Turkewitz & Kenny (1982) / Gottlieb (1971) idea that the staggered,
fixed order of sensory onset in utero (touch/vestibular -> chemical ->
audition -> vision) reduces competition between senses, against simultaneous
onset and the reversed order. Kleinman, Achille, Soatto et al. (ICLR 2023)
showed critical periods for multisensory integration in deep nets, which this
probes from the onset-order side.

The fused model: per-sense input adapters -> sum (absent senses contribute 0)
-> one GRU core -> per-sense output adapters predicting each sense's next frame.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from . import world
from .age import perceive
from .models import E
from .run import BURN, T, _probe

S = world.SENSES


class Fused(nn.Module):
    def __init__(self, H=96):
        super().__init__()
        self.enc = nn.ModuleDict({s: nn.Sequential(nn.Linear(world.DIMS[s], E), nn.GELU()) for s in S})
        self.core = nn.GRU(E, H, batch_first=True)
        self.dec = nn.ModuleDict({s: nn.Linear(H, world.DIMS[s]) for s in S})

    def forward(self, xs, on):
        z = sum(self.enc[s](xs[s]) * float(on[s]) for s in S)
        h, _ = self.core(z)
        return {s: self.dec[s](h) for s in S}, h


def make_batch(age, B, rng, violation=None, t_v=12, with_clean=False):
    lat = world.sample_latents(B, T, rng, violation=violation, t_v=t_v)
    R = {s: world.render(s, lat) for s in S}
    xs = {s: torch.from_numpy(perceive(s, R[s], age, rng)) for s in S}
    if with_clean:
        return xs, lat, {s: torch.from_numpy(perceive(s, R[s], age, rng, noise_on=False)) for s in S}
    return xs, lat


SCHEDULES = {
    # fraction of training at which each sense switches on
    "simultaneous": {"touch": 0.0, "audio": 0.0, "vision": 0.0},
    "biological":   {"touch": 0.0, "audio": 0.25, "vision": 0.5},
    "reversed":     {"vision": 0.0, "audio": 0.25, "touch": 0.5},
}


def train_fused(schedule, steps, seed, age=12, B=64):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = Fused()
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    onset = SCHEDULES[schedule]
    for st in range(steps):
        frac = st / steps
        on = {s: frac >= onset[s] for s in S}
        xs, _ = make_batch(age, B, rng)
        preds, _ = m({s: xs[s][:, :-1] for s in S}, on)
        # a model only learns to predict senses it currently receives
        loss = sum(nn.functional.mse_loss(preds[s], xs[s][:, 1:]) for s in S if on[s])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
    return m


@torch.no_grad()
def eval_fused(m, seed=999, N=512, age=12):
    """Probe the shared state with each sense alone and with all senses."""
    out = {}
    rng = np.random.default_rng(seed)
    xs, lat, clean = make_batch(age, N, rng, with_clean=True)
    lat_in = {k: v[:, :-1] for k, v in lat.items()}
    conds = {f"only_{s}": {t: t == s for t in S} for s in S}
    conds["all"] = {s: True for s in S}
    for name, on in conds.items():
        preds, h = m({s: xs[s][:, :-1] for s in S}, on)
        mask = np.zeros_like(lat_in["x"], dtype=bool)
        if name.startswith("only_"):
            mask = world.unobserved_mask(name[5:], lat_in)
        p = _probe(h.numpy(), lat_in, mask, np.random.default_rng(seed + 1))
        out[f"{name}_pos_r2"] = p["pos_r2"]
        out[f"{name}_id_acc"] = p["id_acc"]
        if name.startswith("only_"):
            s = name[5:]
            c = clean[s]
            err = ((preds[s] - c[:, 1:]) ** 2).mean(-1)[:, BURN:].mean()
            base = ((c[:, 1:] - c[:, 1:].mean((0, 1))) ** 2).mean(-1)[:, BURN:].mean()
            out[f"{name}_skill"] = float(1 - err / base)
    # cross-modal: predict vision's next frame when only audio+touch are on
    on = {"vision": False, "audio": True, "touch": True}
    preds, _ = m({s: xs[s][:, :-1] for s in S}, on)
    err = ((preds["vision"] - xs["vision"][:, 1:]) ** 2).mean()
    mean_img = ((xs["vision"][:, 1:] - xs["vision"][:, 1:].mean((0, 1))) ** 2).mean()
    out["xmodal_vision_from_audio_touch"] = float(1 - err / mean_img)
    return out
