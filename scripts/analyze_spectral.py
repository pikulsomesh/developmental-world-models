"""Spectral account of learning time under sensory immaturity (linear read-out, exact gradient flow).

For input x (flattened, centred, filtered by the age-a front end) and target y (toy position, standardised),
gradient flow on squared error with step eta gives, in the eigenbasis of the input covariance (lam_i),
    R2(t) = sum_i  c_i^2 / (lam_i * var_y)  *  (1 - exp(-2 eta lam_i t)),
so mode i is learned on the time scale 1/(2 eta lam_i). The front end changes lam_i and c_i; this script
measures how, and the time to reach half of the linear ceiling at each age. Writes results/spectral.json.
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dwm import env, senses, capability as cap

AGES = [0, 3, 6, 12]
N_EP = int(os.environ.get("N_EP", 800))


def stats(age, frames, pos, sigma=None, gain=None):
    x = torch.from_numpy(frames)
    if sigma is None and gain is None:
        x = senses.vision(x, age)
    else:  # controlled sweep: blur only or contrast only
        if sigma is not None:
            x = senses.blur2d(x, sigma)
        if gain is not None:
            m = x.mean(dim=(1, 2, 3), keepdim=True); x = m + (x - m) * gain
    X = x.reshape(len(x), -1).numpy().astype(np.float64)
    X -= X.mean(0)
    y = (pos - pos.mean(0)) / pos.std(0)
    n = len(X)
    S = X.T @ X / n
    c = X.T @ y / n                       # (D, 2)
    lam, V = np.linalg.eigh(S)
    lam = np.clip(lam, 0, None)
    ct = V.T @ c                           # (D, 2) components along eigenvectors
    keep = lam > 1e-9 * lam.max()
    w = (ct[keep] ** 2).sum(1) / lam[keep]  # per-mode contribution to the ceiling (sum over 2 targets)
    ceiling = float(w.sum() / 2)           # mean R2 over x,y targets
    return lam[keep], w / 2, ceiling


def r2_curve(lam, w, eta, ts):
    return [float((w * (1 - np.exp(-2 * eta * lam * t))).sum()) for t in ts]


def t_half(lam, w, eta, ceiling):
    lo, hi = 1.0, 1e9
    for _ in range(80):
        mid = (lo * hi) ** 0.5
        if (w * (1 - np.exp(-2 * eta * lam * mid))).sum() >= 0.5 * ceiling:
            hi = mid
        else:
            lo = mid
    return hi


def main():
    d = env.generate(N_EP, seed=11, occluder_prob=0.0)
    T = d["vision"].shape[1]
    frames = d["vision"].reshape(-1, 3, 32, 32)
    pos = d["obj_pos"].reshape(-1, 2) / 31.0
    vis = d["visible"].reshape(-1) > 0.5
    frames, pos = frames[vis], pos[vis]
    sel = np.random.default_rng(0).choice(len(frames), min(6000, len(frames)), replace=False)
    frames, pos = frames[sel], pos[sel]

    out = {"n_frames": int(len(frames)), "ages": {}}
    lam_ad, w_ad, ceil_ad = stats(12, frames, pos)
    eta = 1.0 / lam_ad.max()               # step size fixed by the adult input (stable for every age)
    for a in AGES:
        lam, w, ceil = stats(a, frames, pos)
        p = cap.vision_params(a)
        th = t_half(lam, w, eta, ceil)
        ts = np.logspace(0, 4.5, 40)
        out["ages"][str(a)] = {
            "spectrum": [float(v) for v in np.sort(lam)[::-1][:80]],
            "curve_t": [float(t) for t in ts], "curve_r2": r2_curve(lam, w, eta, ts),
            "blur_sigma_px": p["blur_sigma_px"], "contrast_gain": p["contrast_gain"],
            "total_variance": float(lam.sum()), "lambda_max": float(lam.max()),
            "linear_ceiling_r2": ceil, "t_half": th,
            "task_weighted_lambda": float((w * lam).sum() / w.sum()),
            "eff_rank": float(lam.sum() ** 2 / (lam ** 2).sum()),
        }
    base = out["ages"]["12"]["t_half"]
    for a in AGES:
        out["ages"][str(a)]["t_half_rel"] = out["ages"][str(a)]["t_half"] / base
    # controlled sweeps at adult quality otherwise
    sw = {"blur": [], "contrast": []}
    for s in [0.0, 0.5, 1.0, 1.5, 2.14, 3.0]:
        lam, w, ceil = stats(12, frames, pos, sigma=s if s > 0 else 0.0, gain=None)
        sw["blur"].append({"sigma": s, "t_half_rel": t_half(lam, w, eta, ceil) / base, "ceiling": ceil,
                           "task_weighted_lambda": float((w * lam).sum() / w.sum())})
    for g in [1.0, 0.7, 0.5, 0.3, 0.15]:
        lam, w, ceil = stats(12, frames, pos, sigma=0.0, gain=g)
        sw["contrast"].append({"gain": g, "t_half_rel": t_half(lam, w, eta, ceil) / base,
                               "predicted_1_over_g2": 1 / g ** 2, "ceiling": ceil})
    out["sweeps"] = sw
    # sanity: closed form vs numerical gradient descent at age 0
    x = senses.vision(torch.from_numpy(frames), 0).reshape(len(frames), -1).numpy().astype(np.float64)
    x -= x.mean(0); y = (pos - pos.mean(0)) / pos.std(0)
    lam, w, ceil = stats(0, frames, pos)
    W = np.zeros((x.shape[1], 2)); ts = [50, 200, 800, 3200]; chk = []
    for t in range(1, ts[-1] + 1):
        W += eta * (x.T @ (y - x @ W)) / len(x)
        if t in ts:
            pred = 1 - ((y - x @ W) ** 2).mean(0)
            chk.append({"t": t, "gd_r2": float(pred.mean()), "closed_form": r2_curve(lam, w, eta, [t])[0]})
    out["gd_check_age0"] = chk
    json.dump(out, open("results/spectral.json", "w"), indent=1)
    print(json.dumps({a: {k: round(v, 4) for k, v in out["ages"][a].items()} for a in out["ages"]}, indent=1))
    print(out["sweeps"]); print(chk)


if __name__ == "__main__":
    main()
