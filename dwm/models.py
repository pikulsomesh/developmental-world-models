"""Sense-specific world models that can grow, and a fusion model that combines them.

SenseWorldModel(sense, width, window):
    encoder   o_t -> z_t                      (sense-specific)
    dynamics  [z_t, z_{t-1}, ..., z_{t-k+1}], a_t -> z_{t+1}   (residual MLP)
    decoder   z -> o                         (sense-specific)

grow(width, window) enlarges the model *function-preservingly*: new hidden units get
random incoming weights and zero outgoing weights, and new (older) memory slots get
zero input weights. The grown model computes exactly what the younger one did, then
keeps learning. Latent size is fixed per sense so the fusion interface stays stable.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT = {"vision": 32, "audio": 16, "touch": 16, "proprio": 8}
OBS_DIM = {"audio": 64, "touch": 64, "proprio": 4}
ACTION_DIM = 2


# ------------------------------------------------------------------ growth helpers
def _new_like(layer, **kw):
    if isinstance(layer, nn.Linear):
        return nn.Linear(kw.get("in_f", layer.in_features), kw.get("out_f", layer.out_features))
    if isinstance(layer, nn.Conv2d):
        return nn.Conv2d(kw.get("in_f", layer.in_channels), kw.get("out_f", layer.out_channels),
                         layer.kernel_size, layer.stride, layer.padding)
    if isinstance(layer, nn.ConvTranspose2d):
        return nn.ConvTranspose2d(kw.get("in_f", layer.in_channels), kw.get("out_f", layer.out_channels),
                                  layer.kernel_size, layer.stride, layer.padding)
    raise TypeError(layer)


def _out_axis(layer):
    return 1 if isinstance(layer, nn.ConvTranspose2d) else 0


def expand_out(layer, n_new):
    """Add n_new output units with fresh random incoming weights."""
    if n_new <= 0:
        return layer
    out_f = (layer.out_features if isinstance(layer, nn.Linear) else layer.out_channels) + n_new
    new = _new_like(layer, out_f=out_f)
    ax = _out_axis(layer)
    with torch.no_grad():
        idx = [slice(None)] * new.weight.dim()
        idx[ax] = slice(0, layer.weight.shape[ax])
        new.weight[tuple(idx)] = layer.weight
        new.bias[: layer.bias.shape[0]] = layer.bias
        new.bias[layer.bias.shape[0]:] = 0
    return new


def expand_in(layer, n_new):
    """Add n_new input units with zero weights (function-preserving)."""
    if n_new <= 0:
        return layer
    in_f = (layer.in_features if isinstance(layer, nn.Linear) else layer.in_channels) + n_new
    new = _new_like(layer, in_f=in_f)
    ax = 1 - _out_axis(layer)
    with torch.no_grad():
        new.weight.zero_()
        idx = [slice(None)] * new.weight.dim()
        idx[ax] = slice(0, layer.weight.shape[ax])
        new.weight[tuple(idx)] = layer.weight
        new.bias.copy_(layer.bias)
    return new


# ------------------------------------------------------------------ encoders / decoders
class VisionCodec(nn.Module):
    """32x32x3 <-> latent. Hidden sizes scale with width.

    The decoder is a spatial-broadcast decoder (Watters et al., 2019): the latent is
    tiled over an 8x8 grid with coordinate channels, which makes "where is the toy"
    easy to express and learn on a CPU budget.
    """

    def __init__(self, width, latent):
        super().__init__()
        self.c, self.h = 8 * width, 64 * width
        c, h = self.c, self.h
        self.enc = nn.ModuleList([nn.Conv2d(5, c, 4, 2, 1), nn.Conv2d(c, 2 * c, 4, 2, 1),
                                  nn.Conv2d(2 * c, 2 * c, 4, 2, 1), nn.Linear(2 * c * 16, h), nn.Linear(h, latent)])
        self.dec = nn.ModuleList([nn.Conv2d(latent + 2, 2 * c, 3, 1, 1), nn.Conv2d(2 * c, 2 * c, 3, 1, 1),
                                  nn.ConvTranspose2d(2 * c, c, 4, 2, 1), nn.ConvTranspose2d(c, 3, 4, 2, 1)])

    @staticmethod
    def _coords(n, size, device):
        g = torch.linspace(-1, 1, size, device=device)
        yy, xx = torch.meshgrid(g, g, indexing="ij")
        return torch.stack([xx, yy]).expand(n, 2, size, size)

    def encode(self, x):
        e = self.enc
        x = torch.cat([x, self._coords(x.shape[0], x.shape[-1], x.device)], dim=1)
        x = F.relu(e[0](x)); x = F.relu(e[1](x)); x = F.relu(e[2](x))
        return e[4](F.relu(e[3](x.flatten(1))))

    def decode(self, z):
        d = self.dec
        x = z[:, :, None, None].expand(-1, -1, 8, 8)
        x = torch.cat([x, self._coords(z.shape[0], 8, z.device)], dim=1)
        x = F.relu(d[0](x)); x = F.relu(d[1](x)); x = F.relu(d[2](x))
        return torch.sigmoid(d[3](x))

    def grow(self, width):
        c2, h2 = 8 * width, 64 * width
        dc, dh = c2 - self.c, h2 - self.h
        e, d = self.enc, self.dec
        e[0] = expand_out(e[0], dc)
        e[1] = expand_out(expand_in(e[1], dc), 2 * dc)
        e[2] = expand_out(expand_in(e[2], 2 * dc), 2 * dc)
        e[3] = expand_out(expand_in(e[3], 2 * dc * 16), dh)  # channel-major flatten: new block at end
        e[4] = expand_in(e[4], dh)
        d[0] = expand_out(d[0], 2 * dc)
        d[1] = expand_out(expand_in(d[1], 2 * dc), 2 * dc)
        d[2] = expand_out(expand_in(d[2], 2 * dc), dc)
        d[3] = expand_in(d[3], dc)
        self.c, self.h = c2, h2


class VectorCodec(nn.Module):
    def __init__(self, width, latent, obs_dim):
        super().__init__()
        self.h = 32 * width
        h = self.h
        self.enc = nn.ModuleList([nn.Linear(obs_dim, h), nn.Linear(h, h), nn.Linear(h, latent)])
        self.dec = nn.ModuleList([nn.Linear(latent, h), nn.Linear(h, h), nn.Linear(h, obs_dim)])

    def encode(self, x):
        e = self.enc
        return e[2](F.relu(e[1](F.relu(e[0](x)))))

    def decode(self, z):
        d = self.dec
        return d[2](F.relu(d[1](F.relu(d[0](z)))))

    def grow(self, width):
        dh = 32 * width - self.h
        for m in (self.enc, self.dec):
            m[0] = expand_out(m[0], dh)
            m[1] = expand_out(expand_in(m[1], dh), dh)
            m[2] = expand_in(m[2], dh)
        self.h += dh


class Dynamics(nn.Module):
    """Residual MLP over a window of k latents plus the current action."""

    def __init__(self, width, window, latent):
        super().__init__()
        self.latent, self.window, self.h = latent, window, 64 * width
        self.net = nn.ModuleList([nn.Linear(window * latent + ACTION_DIM, self.h), nn.Linear(self.h, self.h),
                                  nn.Linear(self.h, latent)])

    def forward(self, zwin, a):
        # zwin: (N, k, D) newest first
        n = self.net
        x = torch.cat([a, zwin.flatten(1)], dim=1)
        return zwin[:, 0] + n[2](F.relu(n[1](F.relu(n[0](x)))))

    def grow(self, width, window):
        dh = 64 * width - self.h
        dk = window - self.window
        n = self.net
        # input layout is [action, z_t, z_{t-1}, ...]: older slots are appended at the end
        n[0] = expand_out(expand_in(n[0], dk * self.latent), dh)
        n[1] = expand_out(expand_in(n[1], dh), dh)
        n[2] = expand_in(n[2], dh)
        self.h, self.window = self.h + dh, window


def windows(z, k):
    """z: (B, T, D) -> (B, T, k, D), newest first, padded by repeating frame 0."""
    B, T, D = z.shape
    pad = z[:, :1].expand(B, k - 1, D)
    zp = torch.cat([pad, z], dim=1)
    idx = torch.arange(T).view(T, 1) + (k - 1) - torch.arange(k).view(1, k)
    return zp[:, idx]


class SenseWorldModel(nn.Module):
    def __init__(self, sense, width=1, window=1):
        super().__init__()
        self.sense, self.width, self.window = sense, width, window
        D = LATENT[sense]
        self.codec = VisionCodec(width, D) if sense == "vision" else VectorCodec(width, D, OBS_DIM[sense])
        self.dyn = Dynamics(width, window, D)

    def encode_seq(self, o):
        B, T = o.shape[:2]
        return self.codec.encode(o.reshape(B * T, *o.shape[2:])).view(B, T, -1)

    def decode_seq(self, z):
        B, T = z.shape[:2]
        out = self.codec.decode(z.reshape(B * T, -1))
        return out.view(B, T, *out.shape[1:])

    def forward(self, o, a):
        """Returns latents z (B,T,D) and one-step predictions zhat for t+1 (B,T-1,D)."""
        z = self.encode_seq(o)
        B, T, D = z.shape
        zw = windows(z, self.window)[:, :-1]
        zhat = self.dyn(zw.reshape(-1, self.window, D), a[:, :-1].reshape(-1, ACTION_DIM)).view(B, T - 1, D)
        return z, zhat

    def loss(self, o, a):
        z, zhat = self(o, a)
        w = self.salience(o)
        recon = _wmse(self.decode_seq(z), o, w)
        pred = _wmse(self.decode_seq(zhat), o[:, 1:], None if w is None else w[:, 1:])
        latent = F.mse_loss(zhat, z[:, 1:].detach())
        return recon + pred + 0.1 * latent, {"recon": recon.item(), "pred": pred.item()}

    def salience(self, o):
        """Per-element loss weights. Vision up-weights what changes between frames
        (infants orient to motion); without it the small moving toy is ignored by MSE."""
        if self.sense != "vision":
            return None
        d = (o[:, 1:] - o[:, :-1]).abs().amax(dim=2, keepdim=True)
        d = torch.cat([d[:, :1], d], dim=1)
        return 1.0 + 20.0 * d

    def grow(self, width, window):
        if width < self.width or window < self.window:
            raise ValueError("models only grow")
        self.codec.grow(width)
        self.dyn.grow(width, window)
        self.width, self.window = width, window
        return self

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def _wmse(pred, target, w):
    if w is None:
        return F.mse_loss(pred, target)
    return ((pred - target) ** 2 * w).mean() / w.mean()


# ------------------------------------------------------------------ fusion
class FusionWorldModel(nn.Module):
    """Combines frozen sense-specific encoders into one multisensory world model.

    Each sense latent is projected to a shared token; a modality-masked mean plus
    an MLP gives a fused state f_t. A windowed dynamics model predicts f_{t+1}, and
    per-sense read-out heads reconstruct every sense latent from f (cross-modal
    filling-in). Modality dropout during training teaches the fused state to carry
    information about senses that are momentarily missing (e.g. an occluded toy).
    """

    def __init__(self, senses, window=6, dim=64):
        super().__init__()
        self.senses, self.window, self.dim = list(senses), window, dim
        self.proj = nn.ModuleDict({s: nn.Linear(LATENT[s], dim) for s in self.senses})
        self.modality_emb = nn.ParameterDict({s: nn.Parameter(torch.zeros(dim)) for s in self.senses})
        self.mix = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.dyn = Dynamics(width=2, window=window, latent=dim)
        self.read = nn.ModuleDict({s: nn.Linear(dim, LATENT[s]) for s in self.senses})

    def fuse(self, zs, mask):
        """zs: {sense: (B,T,D_s)}; mask: {sense: (B,T,1)} 1 = present."""
        tok = 0
        cnt = 0
        for s in self.senses:
            tok = tok + mask[s] * (self.proj[s](zs[s]) + self.modality_emb[s])
            cnt = cnt + mask[s]
        return self.mix(tok / cnt.clamp(min=1.0))

    def forward(self, zs, mask, a):
        f = self.fuse(zs, mask)
        B, T, D = f.shape
        fw = windows(f, self.window)
        # recurrent-free state: carry information through time via the window
        state = self.dyn(fw.reshape(-1, self.window, D), a.reshape(-1, ACTION_DIM)).view(B, T, D)
        return f, state  # state at t is the prediction of the fused state at t+1

    def loss(self, zs, mask, a):
        f, state = self(zs, mask, a)
        loss = 0.0
        for s in self.senses:
            tgt = zs[s].detach()
            loss = loss + F.mse_loss(self.read[s](state[:, :-1]), tgt[:, 1:]) / tgt.var().clamp(min=1e-3)
            loss = loss + F.mse_loss(self.read[s](f), tgt) / tgt.var().clamp(min=1e-3)
        return loss / len(self.senses)


def clone(m):
    return copy.deepcopy(m)
