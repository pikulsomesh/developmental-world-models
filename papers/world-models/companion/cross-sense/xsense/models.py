"""Five world-model families with a matched *core* parameter budget.

Every model maps a sequence of per-sense inputs x_t (dim D) to a prediction of
the next input (or next latent, for JEPA) and exposes its hidden state for
probing. Each has a sense-specific input adapter (D -> E) and, except JEPA, an
output adapter (H -> D). Those adapters scale with the sense's input size, the
way primary sensory cortices differ in size, so the budget we match across
senses and families is the *core* (everything between the adapters).

  ff    : no memory; MLP from the current frame (a "reflexive" baseline)
  gru   : recurrent world model (the proposal's LSTM memory, simplified)
  tf    : causal transformer with a limited context window (working memory span)
  pc    : predictive-coding RNN: the recurrent core only receives the
          prediction error x_t - x_hat_t, not the input itself
  jepa  : joint-embedding predictive model: predicts the next *latent* of an EMA
          target encoder, never reconstructs the input
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

E = 32  # adapter output width, fixed for all senses


def n_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class Base(nn.Module):
    latent_target = False

    def core_params(self):
        return n_params(self.core)


class FF(Base):
    def __init__(self, D, H):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(D, E), nn.GELU())
        self.core = nn.Sequential(nn.Linear(E, H), nn.GELU(), nn.Linear(H, H), nn.GELU())
        self.dec = nn.Linear(H, D)

    def forward(self, x):
        h = self.core(self.enc(x))
        return self.dec(h), h


class GRU(Base):
    def __init__(self, D, H):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(D, E), nn.GELU())
        self.core = nn.GRU(E, H, batch_first=True)
        self.dec = nn.Linear(H, D)

    def forward(self, x):
        h, _ = self.core(self.enc(x))
        return self.dec(h), h


class _TFCore(nn.Module):
    def __init__(self, H, T_max, layers=2):
        super().__init__()
        self.inp = nn.Linear(E, H)
        self.pos = nn.Parameter(torch.zeros(1, T_max, H))
        layer = nn.TransformerEncoderLayer(H, nhead=4 if H % 4 == 0 else 1, dim_feedforward=2 * H,
                                           dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class TF(Base):
    def __init__(self, D, H, T_max=32, window=8):
        super().__init__()
        self.window = window
        self.enc = nn.Sequential(nn.Linear(D, E), nn.GELU())
        self.core = _TFCore(H, T_max)
        self.dec = nn.Linear(H, D)

    def forward(self, x):
        T = x.shape[1]
        z = self.core.inp(self.enc(x)) + self.core.pos[:, :T]
        i = torch.arange(T)
        mask = (i[None, :] > i[:, None]) | (i[:, None] - i[None, :] >= self.window)
        h = self.core.tf(z, mask=mask)
        return self.dec(h), h


class PC(Base):
    def __init__(self, D, H):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(D, E), nn.GELU())
        self.core = nn.GRUCell(E, H)
        self.dec = nn.Linear(H, D)
        self.H = H

    def forward(self, x):
        B, T, D = x.shape
        h = x.new_zeros(B, self.H)
        xhat = x.new_zeros(B, D)
        preds, hs = [], []
        for t in range(T):
            err = x[:, t] - xhat                 # only the surprise is passed upward
            h = self.core(self.enc(err), h)
            xhat = self.dec(h)
            preds.append(xhat)
            hs.append(h)
        return torch.stack(preds, 1), torch.stack(hs, 1)


class JEPA(Base):
    latent_target = True

    def __init__(self, D, H, ema=0.98):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(D, E), nn.GELU(), nn.Linear(E, E))
        self.core = nn.ModuleDict({"rnn": nn.GRU(E, H, batch_first=True), "head": nn.Linear(H, E)})
        self.target = copy.deepcopy(self.enc)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.ema = ema

    def forward(self, x):
        z = self.enc(x)
        h, _ = self.core["rnn"](z)
        return self.core["head"](h), h

    @torch.no_grad()
    def target_latent(self, x):
        return self.target(x)

    @torch.no_grad()
    def update_target(self):
        for pt, po in zip(self.target.parameters(), self.enc.parameters()):
            pt.mul_(self.ema).add_(po.detach(), alpha=1 - self.ema)


FAMILIES = {"ff": FF, "gru": GRU, "tf": TF, "pc": PC, "jepa": JEPA}


def build(family, D, core_budget, seed=0, **kw):
    """Pick the hidden width H so the core parameter count is closest to budget."""
    cls = FAMILIES[family]
    best = None
    for H in range(4, 513, 4):
        torch.manual_seed(0)
        m = cls(D, H, **kw)
        gap = abs(m.core_params() - core_budget)
        if best is None or gap < best[0]:
            best = (gap, H)
        if m.core_params() > core_budget * 1.3:
            break
    torch.manual_seed(seed)
    return cls(D, best[1], **kw)


def step_loss(model, x):
    """Next-step prediction loss on a batch x of shape (B, T, D)."""
    pred, _ = model(x[:, :-1])
    if model.latent_target:
        tgt = model.target_latent(x[:, 1:])
        loss = F.mse_loss(pred, tgt)
        z = model.enc(x.reshape(-1, x.shape[-1]))
        std = torch.sqrt(z.var(0) + 1e-4)
        loss = loss + F.relu(1.0 - std).mean()          # variance term against collapse
        return loss
    return F.mse_loss(pred, x[:, 1:])


@torch.no_grad()
def per_step_error(model, x, clean=None):
    """Per-sample, per-step prediction error (B, T-1) and the predict-the-mean baseline.

    If `clean` (same input without sensor noise) is given, both errors are scored
    against it, so the skill score is not dominated by the unpredictable noise
    floor of immature senses.
    """
    pred, h = model(x[:, :-1])
    c = x if clean is None else clean
    if model.latent_target:
        tgt = model.target_latent(c)
        err = ((pred - tgt[:, 1:]) ** 2).mean(-1)
        base = ((tgt[:, 1:] - tgt[:, 1:].mean((0, 1))) ** 2).mean(-1)
    else:
        err = ((pred - c[:, 1:]) ** 2).mean(-1)
        base = ((c[:, 1:] - c[:, 1:].mean((0, 1))) ** 2).mean(-1)
    return err, base, h
