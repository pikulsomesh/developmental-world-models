"""Combining the per-sense world models into one multisensory world model."""
from __future__ import annotations

import numpy as np
import torch

from . import capability as cap
from .models import FusionWorldModel, LATENT
from .train import frame_features, latents, probe

SENSES = ["vision", "audio", "touch", "proprio"]


@torch.no_grad()
def sense_latents(models, data, age):
    return {s: latents(models[s], data, age)[0] for s in SENSES}


def _random_mask(B, T, gen, p_drop=0.3):
    m = {s: (torch.rand(B, T, 1, generator=gen) > p_drop).float() for s in SENSES}
    none = sum(m.values()) == 0
    m["proprio"] = torch.maximum(m["proprio"], none.float())  # always keep at least one sense
    return m


def train_fusion(zs, action, window, steps, seed=0, bs=64, lr=2e-3, log=None):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = FusionWorldModel(SENSES, window=window)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = action.shape[0]
    for step in range(steps):
        idx = torch.randint(0, N, (bs,), generator=gen)
        z = {s: zs[s][idx] for s in SENSES}
        mask = _random_mask(bs, action.shape[1], gen)
        loss = model.loss(z, mask, action[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log and (step + 1) % 1000 == 0:
            log("fusion step", step + 1, round(loss.item(), 4))
    return model


@torch.no_grad()
def fused_features(model, zs, action, drop=()):
    B, T = action.shape[:2]
    mask = {s: torch.zeros(B, T, 1) if s in drop else torch.ones(B, T, 1) for s in SENSES}
    f, state = model(zs, mask, action)
    return f, state


def _hidden_split(data):
    vis = data["visible"].numpy()
    present = data["present"].numpy()
    return (vis == 0) & (present > 0)


def run_fusion(snapshots, train, test, voe_pos, seed=0, steps=3000, log=None):
    """For each stage, fuse that stage's sense models and measure what fusion adds.

    snapshots: {age: {sense: SenseWorldModel}}  (grown models frozen at each stage)

    1. Cross-modal fill-in: probe toy position/identity from the fused state when
       vision is switched off entirely, vs from vision alone and from all senses.
    2. Occluded tracking: on side-view occlusion episodes, probe the toy's position on
       frames where it is fully hidden, from (a) the vision model's memory window,
       (b) audio alone, (c) the fused world model.
    """
    results = {}
    for age, models in snapshots.items():
        k = cap.capacity(age)["window"]
        ztr = sense_latents(models, train, age)
        zte = sense_latents(models, test, age)
        fus = train_fusion(ztr, train["action"], window=k, steps=steps, seed=seed, log=log)
        fus.eval()

        # --- 1. cross-modal fill-in (ordinary scenes, toy present) -------------------
        N, T = train["present"].shape
        typ_tr = train["obj_type"].view(-1, 1).expand(-1, T).reshape(-1).numpy()
        typ_te = test["obj_type"].view(-1, 1).expand(-1, test["present"].shape[1]).reshape(-1).numpy()
        pos_tr, pos_te = train["obj_pos"].reshape(-1, 2).numpy(), test["obj_pos"].reshape(-1, 2).numpy()
        m_tr = train["present"].reshape(-1).numpy() > 0
        m_te = test["present"].reshape(-1).numpy() > 0
        fill = {}
        feats = {
            "vision_only": (frame_features(ztr["vision"]), frame_features(zte["vision"])),
            "audio_only": (frame_features(ztr["audio"]), frame_features(zte["audio"])),
        }
        for name, drop in [("fused_all", ()), ("fused_no_vision", ("vision",))]:
            ftr, _ = fused_features(fus, ztr, train["action"], drop)
            fte, _ = fused_features(fus, zte, test["action"], drop)
            feats[name] = (frame_features(ftr), frame_features(fte))
        for name, (a, b) in feats.items():
            fill[name] = probe(a, pos_tr, typ_tr, b, pos_te, typ_te, m_tr, m_te)

        # --- 2. occluded tracking ------------------------------------------------------
        zv = sense_latents(models, voe_pos, age)
        n = voe_pos["present"].shape[0]
        half = n // 2
        hidden = _hidden_split(voe_pos)
        pos = voe_pos["obj_pos"].numpy()
        typ = voe_pos["obj_type"].view(-1, 1).expand(-1, hidden.shape[1]).numpy()
        _, state = fused_features(fus, zv, voe_pos["action"])
        # the fused state at t-1 is the model's belief about t; also give it f_t
        f_now, _ = fused_features(fus, zv, voe_pos["action"])
        belief = torch.cat([f_now, torch.cat([f_now[:, :1], state[:, :-1]], 1)], -1)
        cand = {
            "vision_memory": frame_features(zv["vision"], k),
            "audio_only": frame_features(zv["audio"]),
            "fused": frame_features(belief),
        }
        occl = {}
        tr_rows = np.zeros(hidden.shape, bool)
        tr_rows[:half] = True
        for name, X in cand.items():
            h = hidden.reshape(-1)
            trm, tem = h & tr_rows.reshape(-1), h & ~tr_rows.reshape(-1)
            occl[name] = probe(X, pos.reshape(-1, 2), typ.reshape(-1), X, pos.reshape(-1, 2), typ.reshape(-1), trm, tem)
        results[str(age)] = {"window": k, "fill_in": fill, "occluded_tracking": occl,
                             "n_hidden_frames": int(hidden.sum())}
        if log:
            log("fusion age", age, {n: round(v["pos_mae_px"], 2) for n, v in occl.items()},
                {n: round(v["type_acc"], 2) for n, v in fill.items()})
    return results
