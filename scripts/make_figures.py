"""Merge per-condition metrics and draw the figures used in RESULTS.md and the paper.

Usage: python scripts/make_figures.py [--results results]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dwm import senses as S  # noqa: E402
from dwm.env import generate  # noqa: E402

COND_ORDER = ["curriculum_only", "grown", "growth_only", "adult_scratch", "shuffled_order", "reversed_immaturity"]
COND_LABEL = {"grown": "Immature input + growth", "curriculum_only": "Immature input (fixed size)",
              "growth_only": "Growth only", "adult_scratch": "Adult input, fixed size",
              "shuffled_order": "Shuffled input quality", "reversed_immaturity": "Reversed (sharp first)"}
SHORT = {"grown": "immature\n+growth", "curriculum_only": "immature\ninput", "growth_only": "growth\nonly",
         "adult_scratch": "adult\ninput", "shuffled_order": "shuffled", "reversed_immaturity": "reversed"}
COLORS = {"curriculum_only": "#2a78d6", "grown": "#eb6834", "growth_only": "#1baf7a", "adult_scratch": "#eda100",
          "shuffled_order": "#e87ba4", "reversed_immaturity": "#008300"}
SENSE_COLORS = {"vision": "#2a78d6", "audio": "#eb6834", "touch": "#1baf7a", "proprio": "#eda100"}
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e4e4e4"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 5, "savefig.dpi": 160, "savefig.bbox": "tight",
})

# the headline read-out for each sense
HEADLINE = {
    "vision": ("probe_adult", "pos_r2", "toy position R²"),
    "audio": ("probe_adult", "pos_r2", "toy position R² (localisation)"),
    "touch": ("probe_adult", "type_acc", "toy identity acc. (contact)"),
    "proprio": ("probe_adult", "pos_r2", "hand position R²"),
}


def load(results):
    runs = {}
    for path in sorted(glob.glob(os.path.join(results, "*", "seed*", "metrics.json"))):
        cond_dir = path.split(os.sep)[-3]
        seed = path.split(os.sep)[-2]
        with open(path) as f:
            runs[(cond_dir, seed)] = json.load(f)
    return runs


def collect(runs):
    """-> senses[sense][cond] = list over seeds of trajectories; fusion = list over seeds."""
    senses, fusion = {}, []
    for (_, _), m in runs.items():
        for sense, conds in m["senses"].items():
            for cond, traj in conds.items():
                senses.setdefault(sense, {}).setdefault(cond, []).append(traj)
        if m.get("fusion"):
            fusion.append(m["fusion"])
    return senses, fusion


def get(traj_list, stage_idx, key, sub=None):
    vals = []
    for traj in traj_list:
        v = traj[stage_idx][key]
        vals.append(v[sub] if sub else v)
    return np.array(vals, float)


def fig_senses_by_age(out):
    d = generate(60, seed=11)
    # pick the episode/time with the clearest hand contact, so all three senses have content
    tsum = d["touch"].sum(-1)
    e, t = np.unravel_index(int(tsum.argmax()), tsum.shape)
    d = {k: v[e:e + 1] for k, v in d.items()}
    ages = [0, 3, 6, 12]
    fig, ax = plt.subplots(3, 4, figsize=(7.2, 4.6))
    for j, age in enumerate(ages):
        g = torch.Generator().manual_seed(0)
        v = S.apply("vision", torch.from_numpy(d["vision"][0, t:t + 1]), age, g)[0].permute(1, 2, 0)
        a = S.apply("audio", torch.from_numpy(d["audio"][0, t:t + 1]), age, g)[0].view(2, 32)
        tc = S.apply("touch", torch.from_numpy(d["touch"][0, t:t + 1]), age, g)[0].view(8, 8)
        ax[0, j].imshow(v.numpy())
        ax[1, j].imshow(a.numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=0.5)
        ax[2, j].imshow(tc.numpy(), cmap="Greens", vmin=0, vmax=1)
        ax[0, j].set_title(f"{age} months", color=INK)
        for i in range(3):
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([]); ax[i, j].grid(False)
    ax[0, 0].set_ylabel("vision")
    ax[1, 0].set_ylabel("audio\n(L/R × freq)")
    ax[2, 0].set_ylabel("touch")
    fig.suptitle("What each sense delivers at each age (same moment, same scene)", color=INK, fontsize=10)
    fig.savefig(os.path.join(out, "fig_senses_by_age.png"))
    plt.close(fig)


def fig_trajectories(senses, stages, out):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.1))
    for ax, sense in zip(axes, ["vision", "audio", "touch", "proprio"]):
        key, sub, label = HEADLINE[sense]
        for cond in COND_ORDER:
            if cond not in senses.get(sense, {}):
                continue
            tl = senses[sense][cond]
            vals = [get(tl, i, key, sub) for i in range(len(stages))]
            mean = np.array([v.mean() for v in vals])
            sd = np.array([v.std() for v in vals])
            ax.plot(stages, mean, "-o", color=COLORS[cond], label=f"{COND_LABEL[cond]} (n={len(tl)})")
            if len(tl) > 1:
                ax.fill_between(stages, mean - sd, mean + sd, color=COLORS[cond], alpha=0.15, linewidth=0)
        ax.set_title(sense, color=INK)
        if sense == "proprio":
            ax.set_ylim(0.9, 1.0)
        ax.set_xlabel("age stage (months)")
        ax.set_ylabel(label)
        ax.set_xticks(stages)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Developmental trajectories: linear read-out from each sense model's latent (adult-quality test input; "
                 "x = checkpoint)", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_trajectories.png"))
    plt.close(fig)


def fig_robustness(senses, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.0))
    for ax, sense in zip(axes, ["vision", "audio"]):
        conds = [c for c in COND_ORDER if c in senses[sense]]
        x = np.arange(len(conds))
        for j, (key, off, alpha, hatch) in enumerate([("probe_adult", -0.2, 1.0, None),
                                                      ("probe_newborn", 0.2, 0.45, "///")]):
            vals = [get(senses[sense][c], -1, key, "pos_r2") for c in conds]
            ax.bar(x + off, [v.mean() for v in vals], 0.38, color=[COLORS[c] for c in conds], alpha=alpha,
                   hatch=hatch, edgecolor="white", linewidth=1.5,
                   yerr=[v.std() if len(v) > 1 else 0 for v in vals], ecolor=MUTED, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[c] for c in conds], fontsize=7)
        ax.set_title(f"{sense}, 12-month checkpoint: solid = adult input, hatched = newborn-quality input",
                     color=INK, fontsize=8)
        ax.set_ylabel("toy position R² (linear probe)")
        ax.axhline(0, color=MUTED, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_robustness.png"))
    plt.close(fig)


def fig_voe(senses, stages, out):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for cond in COND_ORDER:
        tl = senses["vision"].get(cond)
        if not tl:
            continue
        v = [np.mean([t[i]["voe"]["voe_index"] for t in tl]) for i in range(len(stages))]
        ax.plot(stages, v, "-o", color=COLORS[cond], label=COND_LABEL[cond])
    fv = os.path.join(os.path.dirname(out), "fusion_voe.json")
    if os.path.exists(fv):
        with open(fv) as f:
            d = json.load(f)
        for key, lab, col, ls in [("all", "fused model, all senses", INK, "-"),
                                  ("vision", "fused model, vision channel", INK, ":")]:
            v = [np.mean([d[sd][str(a)][key]["voe_index"] for sd in d]) for a in stages]
            ax.plot(stages, v, ls, marker="s", color=col, label=f"{lab} (n={len(d)})")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(stages)
    ax.set_xlabel("age stage (months)")
    ax.set_ylabel("VoE index (>0 = expects reappearance)")
    ax.set_title("Surprise when a toy vanishes behind an occluder", color=INK, fontsize=9)
    ax.legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.savefig(os.path.join(out, "fig_voe.png"))
    plt.close(fig)


def fig_fusion(fusion, out):
    ages = sorted(fusion[0], key=int)
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.9))
    names = [("vision_memory", "vision (memory window)", "#2a78d6"), ("audio_only", "audio only", "#eb6834"),
             ("fused", "fused world model", "#1baf7a")]
    for name, lab, col in names:
        vals = np.array([[f[a]["occluded_tracking"][name]["x_r2"] for f in fusion] for a in ages])
        axes[0].errorbar([int(a) for a in ages], vals.mean(1), yerr=vals.std(1), fmt="-o", color=col,
                         label=f"{lab} (n={len(fusion)})", capsize=2)
    axes[0].set_title("Tracking a toy while it is fully hidden", color=INK, fontsize=9)
    axes[0].set_ylabel("hidden toy: horizontal position R²")
    axes[0].set_xlabel("age stage (months)")
    axes[0].set_xticks([int(a) for a in ages])
    axes[0].axhline(0, color=MUTED, linewidth=0.8)
    axes[0].legend(fontsize=7)
    names2 = [("vision_only", "vision only", "#2a78d6"), ("fused_all", "fused, all senses", "#1baf7a"),
              ("fused_no_vision", "fused, vision switched off", "#e87ba4")]
    for name, lab, col in names2:
        vals = np.array([[f[a]["fill_in"][name]["pos_r2"] for f in fusion] for a in ages])
        axes[1].errorbar([int(a) for a in ages], vals.mean(1), yerr=vals.std(1), fmt="-o", color=col,
                         label=lab, capsize=2)
    axes[1].set_title("Where is the toy? (ordinary scenes)", color=INK, fontsize=9)
    axes[1].set_ylabel("toy position R²")
    axes[1].set_xlabel("age stage (months)")
    axes[1].set_xticks([int(a) for a in ages])
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_fusion.png"))
    plt.close(fig)


def summary_table(senses, fusion, stages):
    """Mean (and min-max over seeds) of the key numbers, as a JSON-able dict."""
    out = {"senses": {}, "fusion": {}}
    for sense, conds in senses.items():
        out["senses"][sense] = {}
        for cond, tl in conds.items():
            rows = []
            for i, age in enumerate(stages):
                row = {"age": age, "n_seeds": len(tl), "params": tl[0][i]["params"]}
                for key in ["pred_err_own", "pred_err_adult"]:
                    row[key] = float(get(tl, i, key).mean())
                for key in ["probe_adult", "probe_newborn"]:
                    for sub in ["pos_r2", "pos_mae_px", "type_acc"]:
                        row[f"{key}.{sub}"] = float(get(tl, i, key, sub).mean())
                if "voe" in tl[0][i]:
                    row["voe_index"] = float(np.mean([t[i]["voe"]["voe_index"] for t in tl]))
                rows.append(row)
            out["senses"][sense][cond] = rows
    if fusion:
        for a in fusion[0]:
            out["fusion"][a] = {
                sec: {name: {k: float(np.mean([f[a][sec][name][k] for f in fusion])) for k in v}
                      for name, v in fusion[0][a][sec].items()}
                for sec in ["fill_in", "occluded_tracking"]
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    figdir = os.path.join(args.results, "figures")
    os.makedirs(figdir, exist_ok=True)
    fig_senses_by_age(figdir)
    runs = load(args.results)
    if not runs:
        print("no metrics found")
        return
    senses, fusion = collect(runs)
    stages = next(iter(runs.values()))["stages"]
    fig_trajectories(senses, stages, figdir)
    if "vision" in senses:
        fig_robustness(senses, figdir)
        fig_voe(senses, stages, figdir)
    if fusion:
        fig_fusion(fusion, figdir)
    with open(os.path.join(args.results, "summary.json"), "w") as f:
        json.dump(summary_table(senses, fusion, stages), f, indent=1)
    print("wrote", figdir)


if __name__ == "__main__":
    main()
