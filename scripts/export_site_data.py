"""Export the data behind the interactive site in docs/.

  episodes : raw (adult-quality) example episodes as base64 arrays, so the site can
             apply the age filters itself in JavaScript.
  models   : for every regime and checkpoint (seed 0), what the vision model is given,
             what it reconstructs and what it predicts one step ahead, as PNG sprite
             sheets; plus per-frame surprise on the possible / impossible VoE pair.
  results  : headline metrics (mean, s.d., per seed) from results/ for the charts.

Usage: python scripts/export_site_data.py [--parts episodes,models,results]
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dwm import capability as cap  # noqa: E402
from dwm import senses as S  # noqa: E402
from dwm.env import generate  # noqa: E402
from dwm.models import SenseWorldModel  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "docs", "data")
REGIMES = ["curriculum_only", "grown", "growth_only", "adult_scratch", "shuffled_order", "reversed_immaturity"]
TRAIN_AGE = {"curriculum_only": [0, 3, 6, 12], "grown": [0, 3, 6, 12], "growth_only": [12] * 4,
             "adult_scratch": [12] * 4, "shuffled_order": [0, 3, 6, 12], "reversed_immaturity": [12, 6, 3, 0]}


def b64(a, dtype):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


def pick(data, need_contact=True):
    for i in range(len(data["obj_pos"])):
        hidden = (data["visible"][i] < 0.5) & (data["present"][i] > 0.5)
        if hidden.sum() >= 2 and (not need_contact or data["contact"][i].sum() >= 3):
            return i
    return 0


def episode_json(data, i):
    return {
        "T": int(data["vision"].shape[1]),
        "vision": b64(np.round(data["vision"][i] * 255), np.uint8),  # (T,3,32,32)
        "audio": b64(data["audio"][i], np.float32),  # (T,64)
        "touch": b64(data["touch"][i], np.float32),  # (T,64)
        "proprio": b64(data["proprio"][i], np.float32),  # (T,4)
        "obj_pos": data["obj_pos"][i].round(2).tolist(),
        "visible": data["visible"][i].tolist(),
        "present": data["present"][i].tolist(),
        "contact": data["contact"][i].tolist(),
        "type": int(data["obj_type"][i]),
    }


def episodes():
    ordinary = generate(300, seed=777)
    i = pick(ordinary)
    pos, imp = generate(40, seed=778, voe="possible"), generate(40, seed=778, voe="impossible")
    j = pick(pos, need_contact=False)
    out = {"ordinary": episode_json(ordinary, i), "voe_possible": episode_json(pos, j),
           "voe_impossible": episode_json(imp, j)}
    tables = {
        "vision": {"acuity_cpd": cap.VISION_ACUITY_CPD, "contrast_sens": cap.VISION_CONTRAST_SENS,
                   "color_sat": cap.VISION_COLOR_SAT},
        "audio": {"freq_blur": cap.AUDIO_FREQ_BLUR_BINS, "hf_cutoff": cap.AUDIO_HF_CUTOFF,
                  "binaural": cap.AUDIO_BINAURAL, "noise": cap.AUDIO_NOISE},
        "touch": {"blur": cap.TOUCH_BLUR, "noise": cap.TOUCH_NOISE},
        "proprio": {"noise": cap.PROPRIO_NOISE},
        "capacity": {"width": cap.WIDTH, "window": cap.MEMORY_WINDOW},
    }
    with open(os.path.join(OUT, "episodes.json"), "w") as f:
        json.dump({"episodes": out, "tables": tables}, f)
    return ordinary, i, pos, imp, j


def frames_to_row(x):
    """(T,3,32,32) in [0,1] -> (32, T*33-1, 3) uint8 with 1-px gaps."""
    T = x.shape[0]
    row = np.full((32, T * 33 - 1, 3), 255, np.uint8)
    for t in range(T):
        row[:, t * 33 : t * 33 + 32] = np.round(np.clip(x[t].transpose(1, 2, 0), 0, 1) * 255)
    return row


@torch.no_grad()
def models(ordinary, i, pos, imp, j, results):
    os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
    act = torch.from_numpy(ordinary["action"][i : i + 1])
    o = torch.from_numpy(ordinary["vision"][i : i + 1])
    vp, vi = torch.from_numpy(pos["vision"][j : j + 1]), torch.from_numpy(imp["vision"][j : j + 1])
    ap = torch.from_numpy(pos["action"][j : j + 1])
    surprise = {}
    for regime in REGIMES:
        for k, age in enumerate(cap.STAGES):
            p = os.path.join(results, regime, "seed0", f"vision_{regime}_stage{k}.pt")
            if not os.path.exists(p):
                continue
            ck = torch.load(p)
            m = SenseWorldModel("vision", ck["width"], ck["window"])
            m.load_state_dict(ck["state"])
            m.eval()
            a_in = TRAIN_AGE[regime][k]
            of = S.apply("vision", o, a_in)
            z, zhat = m(of, act)
            rec, pred = m.decode_seq(z)[0].numpy(), m.decode_seq(zhat)[0].numpy()
            pred = np.concatenate([np.ones_like(pred[:1]), pred])  # no prediction for frame 0
            sheet = np.concatenate([frames_to_row(of[0].numpy()), np.full((1, rec.shape[0] * 33 - 1, 3), 255, np.uint8),
                                    frames_to_row(rec), np.full((1, rec.shape[0] * 33 - 1, 3), 255, np.uint8),
                                    frames_to_row(pred)])
            Image.fromarray(sheet).save(os.path.join(OUT, "models", f"{regime}_{age}.png"), optimize=True)
            tr = {}
            for name, v in [("possible", vp), ("impossible", vi)]:
                vf = S.apply("vision", v, 12)
                z, zhat = m(vf, ap)
                err = ((m.decode_seq(zhat) - vf[:, 1:]) ** 2).mean(dim=(2, 3, 4))[0].numpy()
                tr[name] = [0.0] + err.round(6).tolist()
            surprise.setdefault(regime, {})[str(age)] = {"input_age": a_in, "params": m.n_params(),
                                                         "width": ck["width"], "window": ck["window"], **tr}
            print(regime, age, "ok", flush=True)
    with open(os.path.join(OUT, "model_outputs.json"), "w") as f:
        json.dump(surprise, f)


def agg(vals):
    v = np.array([x for x in vals if x is not None], float)
    return {"mean": float(v.mean()), "sd": float(v.std()) if len(v) > 1 else 0.0, "n": int(len(v)),
            "seeds": v.round(4).tolist()} if len(v) else None


def results_part(results):
    runs = {}
    for path in glob.glob(os.path.join(results, "*", "seed*", "metrics.json")):
        regime, seed = path.split(os.sep)[-3], path.split(os.sep)[-2]
        runs[(regime, seed)] = json.load(open(path))
    keys = {"pos_r2": ("probe_adult", "pos_r2"), "type_acc": ("probe_adult", "type_acc"),
            "newborn_pos_r2": ("probe_newborn", "pos_r2"), "newborn_type_acc": ("probe_newborn", "type_acc"),
            "voe_index": ("voe", "voe_index"), "pred_err_own": ("pred_err_own", None)}
    senses = {}
    for sense in ["vision", "audio", "touch", "proprio"]:
        for regime in REGIMES:
            per = [r["senses"][sense][regime] for (rg, _), r in runs.items() if rg == regime and sense in r["senses"]
                   and regime in r["senses"][sense]]
            if not per:
                continue
            rows = []
            for k, age in enumerate(cap.STAGES):
                row = {"age": age, "params": per[0][k]["params"]}
                for name, (a, b) in keys.items():
                    vals = [(p[k][a] if b is None else p[k].get(a, {}).get(b)) for p in per if len(p) > k]
                    row[name] = agg(vals)
                rows.append(row)
            senses.setdefault(sense, {})[regime] = rows
    fusion = {}
    for (rg, seed), r in runs.items():
        if rg == "grown" and r.get("fusion"):
            for age, d in r["fusion"].items():
                for grp, sub in d.items():
                    if not isinstance(sub, dict):
                        continue
                    for name, m in sub.items():
                        if not isinstance(m, dict):
                            continue
                        for metric, v in m.items():
                            fusion.setdefault(age, {}).setdefault(grp, {}).setdefault(name, {}).setdefault(metric, []).append(v)
    fusion = {a: {g: {n: {m: agg(v) for m, v in mm.items()} for n, mm in gg.items()} for g, gg in d.items()}
              for a, d in fusion.items()}
    out = {"senses": senses, "fusion": fusion}
    for name in ["fusion_voe.json", "extra.json"]:
        p = os.path.join(results, name)
        if os.path.exists(p):
            out[name.split(".")[0]] = json.load(open(p))
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(out, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--parts", default="episodes,models,results")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("THREADS", "1")))
    os.makedirs(OUT, exist_ok=True)
    parts = args.parts.split(",")
    if "episodes" in parts or "models" in parts:
        ep = episodes()
        if "models" in parts:
            models(*ep, args.results)
    if "results" in parts:
        results_part(args.results)


if __name__ == "__main__":
    main()
