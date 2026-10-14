"""Training and evaluation utilities for sense-specific and fused world models."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import capability as cap
from . import senses
from .models import SenseWorldModel, windows

SENSES = ["vision", "audio", "touch", "proprio"]


def to_torch(data):
    return {k: torch.from_numpy(v) for k, v in data.items()}


def batches(n, bs, gen):
    perm = torch.randperm(n, generator=gen)
    for i in range(0, n - bs + 1, bs):
        yield perm[i : i + bs]


def train_sense(model: SenseWorldModel, data, age, steps, lr=2e-3, bs=32, seed=0, log_every=0, sub_T=10):
    """Train one sense model on observations filtered for `age`."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    obs, act = data[model.sense], data["action"]
    n, step, hist = obs.shape[0], 0, []
    model.train()
    while step < steps:
        for idx in batches(n, bs, gen):
            t0 = int(torch.randint(0, obs.shape[1] - sub_T + 1, (1,), generator=gen))
            o = senses.apply(model.sense, obs[idx, t0 : t0 + sub_T], age, gen)
            loss, parts = model.loss(o, act[idx, t0 : t0 + sub_T])
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if log_every and step % log_every == 0:
                hist.append({"step": step, **parts})
            if step >= steps:
                break
    return hist


@torch.no_grad()
def latents(model, data, age, seed=123, bs=100):
    gen = torch.Generator().manual_seed(seed)
    model.eval()
    zs, zhs = [], []
    obs, act = data[model.sense], data["action"]
    for i in range(0, obs.shape[0], bs):
        o = senses.apply(model.sense, obs[i : i + bs], age, gen)
        z, zh = model(o, act[i : i + bs])
        zs.append(z)
        zhs.append(zh)
    return torch.cat(zs), torch.cat(zhs)


@torch.no_grad()
def prediction_error(model, data, age, seed=123, bs=100, per_step=False):
    """One-step observation prediction MSE (per episode & step if per_step)."""
    gen = torch.Generator().manual_seed(seed)
    model.eval()
    obs, act = data[model.sense], data["action"]
    errs = []
    for i in range(0, obs.shape[0], bs):
        o = senses.apply(model.sense, obs[i : i + bs], age, gen)
        _, zh = model(o, act[i : i + bs])
        pred = model.decode_seq(zh)
        e = (pred - o[:, 1:]) ** 2
        errs.append(e.flatten(2).mean(-1))
    errs = torch.cat(errs)  # (N, T-1)
    return errs if per_step else errs.mean().item()


# ---------------------------------------------------------------- linear probes
def ridge(Xtr, Ytr, Xte, lam=1e-1):
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Xte = np.hstack([Xte, np.ones((len(Xte), 1))])
    Wt = np.linalg.solve(Xtr.T @ Xtr + lam * len(Xtr) * np.eye(Xtr.shape[1]), Xtr.T @ Ytr)
    return Xte @ Wt


def probe(Ztr, ytr_pos, ytr_type, Zte, yte_pos, yte_type, mask_tr=None, mask_te=None):
    """Linear read-out of toy position (R^2) and type (accuracy) from features."""
    if mask_tr is not None:
        Ztr, ytr_pos, ytr_type = Ztr[mask_tr], ytr_pos[mask_tr], ytr_type[mask_tr]
        Zte, yte_pos, yte_type = Zte[mask_te], yte_pos[mask_te], yte_type[mask_te]
    out = {}
    pp = ridge(Ztr, ytr_pos, Zte)
    ss_res = ((pp - yte_pos) ** 2).sum()
    ss_tot = ((yte_pos - yte_pos.mean(0)) ** 2).sum()
    out["pos_r2"] = float(1 - ss_res / ss_tot)
    out["pos_mae_px"] = float(np.abs(pp - yte_pos).mean())
    Y = np.eye(4)[ytr_type]
    pt = ridge(Ztr, Y, Zte)
    out["type_acc"] = float((pt.argmax(1) == yte_type).mean())
    return out


def frame_features(Z, k=1):
    """Flatten (N,T,D) latents (optionally a window of k, newest first) into per-frame rows."""
    Z = torch.as_tensor(Z)
    if k > 1:
        Z = windows(Z, k).flatten(2)
    return Z.reshape(-1, Z.shape[-1]).numpy()


def probe_targets(sense, data):
    """What each sense *should* be able to tell us, and on which frames.

    vision/audio: toy position and identity while the toy exists;
    touch: toy identity (and position) on frames with hand contact;
    proprio: own hand position (it carries no information about the toy).
    """
    N, T = data["present"].shape
    typ = data["obj_type"].view(N, 1).expand(N, T).reshape(-1).numpy()
    if sense == "proprio":
        pos = (data["proprio"][..., :2] * 31.0).reshape(-1, 2).numpy()
        mask = np.ones(N * T, bool)
    else:
        pos = data["obj_pos"].reshape(-1, 2).numpy()
        key = "contact" if sense == "touch" else "present"
        mask = data[key].reshape(-1).numpy() > 0
    return pos, typ, mask


def probe_model(model, train, test, age_train, age_test=None):
    age_test = age_train if age_test is None else age_test
    ztr, _ = latents(model, train, age_train)
    zte, _ = latents(model, test, age_test)
    ptr, ttr, mtr = probe_targets(model.sense, train)
    pte, tte, mte = probe_targets(model.sense, test)
    return probe(frame_features(ztr), ptr, ttr, frame_features(zte), pte, tte, mtr, mte)


def evaluate_sense(model, train, test, own_age):
    """Standard read-outs for one sense model.

    pred_err_own   : one-step prediction error on input at the age it was trained on
    pred_err_adult : the same on adult-quality input
    probe_adult    : linear probes fitted and tested on adult-quality input
    probe_newborn  : probes fitted on adult input, tested on newborn-quality input
                     (robustness to degraded input; Vogelsang et al. 2018)
    """
    return {
        "pred_err_own": prediction_error(model, test, own_age),
        "pred_err_adult": prediction_error(model, test, 12),
        "probe_adult": probe_model(model, train, test, 12),
        "probe_newborn": probe_model(model, train, test, 12, 0),
    }


# ---------------------------------------------------------------- violation of expectation
@torch.no_grad()
def voe_index(model, possible, impossible, age):
    """Surprise at the moment the toy should re-emerge from behind the occluder.

    Surprise = one-step visual prediction error. Index = (S_imp - S_pos)/(S_imp + S_pos):
    > 0 means the model "expects" the toy to reappear (object permanence).
    """
    ep = prediction_error(model, possible, age, per_step=True)
    ei = prediction_error(model, impossible, age, per_step=True)
    vis = possible["visible"].numpy()
    # first visible frame after a hidden stretch, per episode (prediction index = frame-1)
    s_pos, s_imp = [], []
    for i in range(vis.shape[0]):
        hidden = np.where(vis[i] == 0)[0]
        if len(hidden) == 0:
            continue
        t = hidden[-1] + 1
        if t >= vis.shape[1]:
            continue
        sl = slice(t - 1, min(t + 2, vis.shape[1] - 1))  # re-emergence and the next two frames
        s_pos.append(ep[i, sl].mean().item())
        s_imp.append(ei[i, sl].mean().item())
    sp, si = float(np.mean(s_pos)), float(np.mean(s_imp))
    return {"surprise_possible": sp, "surprise_impossible": si, "voe_index": (si - sp) / (si + sp)}
