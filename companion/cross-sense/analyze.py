"""Summarise results/*.json into markdown tables and figures."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

R = os.environ.get("XSENSE_OUT", "results")
FIG = os.path.join(R, "figs")
os.makedirs(FIG, exist_ok=True)
METRICS = ["skill", "pos_r2_hid", "id_acc", "voe_teleport", "voe_identity"]


def load(name):
    p = f"{R}/{name}.json"
    return pd.DataFrame(json.load(open(p))) if os.path.exists(p) else None


def ms(g, cols):
    m, s = g[cols].mean(), g[cols].std()
    return pd.DataFrame({c: [f"{m[c]:.2f} ± {s[c]:.2f}"] for c in cols})


def table(df, keys, cols):
    rows = []
    for k, g in df.groupby(keys, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        r = dict(zip(keys, k))
        for c in cols:
            r[c] = f"{g[c].mean():.2f} ± {g[c].std():.2f}"
        rows.append(r)
    return pd.DataFrame(rows).to_markdown(index=False)


out = []
arch = load("arch")
if arch is not None:
    out.append("## Architectures (core ≈ 20k params, mean ± sd over 3 seeds)\n")
    for age in (0, 12):
        out.append(f"\n### Trained and tested at age {age} mo\n")
        out.append(table(arch[arch.train_age == age], ["sense", "family"], METRICS + ["core_params"]))
        out.append("")
    # family ranking per sense by permanence and teleport VoE at 12 mo
    a12 = arch[arch.train_age == 12].groupby(["sense", "family"])[["pos_r2_hid", "voe_teleport"]].mean()
    out.append("\n### Family rank by teleport VoE at 12 mo (1 = largest surprise)\n")
    out.append(a12["voe_teleport"].groupby(level=0).rank(ascending=False).unstack().to_markdown())
    # heterochrony: newborn/12mo ratio per sense, GRU
    g = arch[arch.family == "gru"].groupby(["sense", "train_age"])[METRICS].mean()
    out.append("\n### GRU at birth vs 12 mo (heterochrony)\n")
    out.append(g.round(2).to_markdown())

scale = load("scale")
if scale is not None:
    out.append("\n## Capacity (GRU, mean over 2 seeds)\n")
    out.append(table(scale, ["sense", "train_age", "budget"], ["skill", "pos_r2_hid", "id_acc", "voe_teleport", "core_params"]))

cur = load("curriculum")
if cur is not None:
    out.append("\n## Input-maturation curricula (GRU 20k, same total experience, mean ± sd over 3 seeds)\n")
    for ea in (12, 0):
        out.append(f"\n### Tested on {ea}-month input\n")
        out.append(table(cur[cur.eval_age == ea], ["sense", "condition"], METRICS))

fu = load("fused")
if fu is not None:
    cols = [c for c in fu.columns if c not in ("schedule", "seed")]
    out.append("\n## Sense onset order in a fused model (mean ± sd over 3 seeds)\n")
    out.append(table(fu, ["schedule"], cols))

open(f"{R}/summary.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))

# figures
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if arch is not None:
        fams = ["ff", "gru", "tf", "pc", "jepa"]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=False)
        for ax, met, title in zip(axes, ["pos_r2_hid", "id_acc", "voe_teleport"],
                                  ["Permanence probe R² (hidden steps)", "Identity probe accuracy", "Teleport VoE (d_z)"]):
            for i, s in enumerate(["vision", "audio", "touch"]):
                sub = arch[(arch.sense == s) & (arch.train_age == 12)].groupby("family")[met]
                m = sub.mean().reindex(fams); e = sub.std().reindex(fams)
                ax.errorbar(np.arange(5) + (i - 1) * 0.2, m, e, fmt="o", label=s, capsize=2)
            ax.set_xticks(range(5)); ax.set_xticklabels(fams); ax.set_title(title, fontsize=10)
        axes[0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(f"{FIG}/arch_12mo.png", dpi=130)
    if scale is not None:
        COLS = {"vision": "#4C72B0", "audio": "#DD8452", "touch": "#55A868"}
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        for ax, met, title in zip(axes, ["skill", "pos_r2_hid", "voe_teleport"],
                                  ["Next-step R²", "Permanence probe R² (clipped at -0.5)", "Teleport VoE (d_z)"]):
            for s in ["vision", "audio", "touch"]:
                for age, ls in ((0, "--"), (12, "-")):
                    sub = scale[(scale.sense == s) & (scale.train_age == age)].groupby("core_params")[met].mean()
                    if sub.notna().any():
                        ax.plot(sub.index, sub.values.clip(-0.5), ls, marker="o", color=COLS[s], label=f"{s} {age} mo")
            ax.set_xscale("log"); ax.set_xlabel("core parameters"); ax.set_title(title, fontsize=10)
        axes[2].legend(fontsize=7, ncol=2)
        fig.suptitle("Capacity sweep (GRU, 2 seeds): dashed = newborn front end, solid = 12-month", fontsize=10)
        fig.tight_layout(); fig.savefig(f"{FIG}/capacity.png", dpi=150)
except Exception as e:  # figures are optional
    print("fig error", e, file=sys.stderr)

# --- additional paper-style figures ---
try:
    import matplotlib.pyplot as plt
    COL = {"vision": "#4C72B0", "audio": "#DD8452", "touch": "#55A868"}
    if arch is not None:
        g = arch[arch.family == "gru"]
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
        for ax, met, title in zip(axes, ["pos_r2_hid", "id_acc", "voe_teleport"],
                                  ["Permanence probe R² (hidden steps)", "Identity probe accuracy", "Teleport VoE (d_z)"]):
            for i, s in enumerate(["touch", "audio", "vision"]):
                for j, age in enumerate((0, 12)):
                    v = g[(g.sense == s) & (g.train_age == age)][met]
                    if v.notna().any():
                        ax.bar(i + (j - 0.5) * 0.38, max(v.mean(), -0.5), 0.36, yerr=v.std(),
                               color=COL[s], alpha=0.45 if age == 0 else 1.0,
                               label=f"{age} mo" if i == 0 else None)
            ax.set_xticks(range(3)); ax.set_xticklabels(["touch", "audio", "vision"]); ax.set_title(title, fontsize=10)
            ax.axhline(0, color="k", lw=0.6)
        axes[0].legend(fontsize=8, title="light = birth")
        axes[0].set_ylim(-0.55, 1)
        fig.suptitle("Same world, same GRU (20k core params): newborn vs 12-month sensory front ends", fontsize=10)
        fig.tight_layout(); fig.savefig(f"{FIG}/heterochrony.png", dpi=150)
    if cur is not None:
        conds = ["mature", "developmental", "newborn", "reversed"]
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
        for ax, met, title in zip(axes, ["skill", "pos_r2_hid", "voe_teleport"],
                                  ["Next-step R² (12-mo input)", "Permanence probe R² (12-mo input)", "Teleport VoE (12-mo input)"]):
            for i, s in enumerate(["vision", "audio", "touch"]):
                for j, c in enumerate(conds):
                    v = cur[(cur.sense == s) & (cur.condition == c) & (cur.eval_age == 12)][met]
                    if v.notna().any():
                        ax.bar(i + (j - 1.5) * 0.2, max(v.mean(), -0.5), 0.19, yerr=v.std(),
                               color=plt.cm.viridis(j / 3), label=c if i == 0 else None)
            ax.set_xticks(range(3)); ax.set_xticklabels(["vision", "audio", "touch"]); ax.set_title(title, fontsize=10)
            ax.axhline(0, color="k", lw=0.6)
        axes[0].set_ylim(-0.55, 1); axes[0].legend(fontsize=7)
        fig.suptitle("Input-maturation curricula at equal total experience (GRU, 3 seeds; bars clipped at -0.5)", fontsize=10)
        fig.tight_layout(); fig.savefig(f"{FIG}/curricula.png", dpi=150)
    if fu is not None:
        sch = ["biological", "simultaneous", "reversed"]
        mets = ["only_vision_skill", "only_audio_skill", "only_touch_skill", "xmodal_vision_from_audio_touch"]
        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        for j, sc in enumerate(sch):
            sub = fu[fu.schedule == sc]
            ax.bar(np.arange(4) + (j - 1) * 0.27, sub[mets].mean(), 0.26, yerr=sub[mets].std(), label=sc,
                   color=plt.cm.Set2(j))
        ax.set_xticks(range(4)); ax.set_xticklabels(["vision alone", "audio alone", "touch alone", "vision from\naudio+touch"])
        ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("next-step R²"); ax.legend(fontsize=8)
        ax.set_title("Order of sense onset in a fused model (3 seeds)", fontsize=10)
        fig.tight_layout(); fig.savefig(f"{FIG}/onset_order.png", dpi=150)
except Exception as e:
    print("fig error", e, file=sys.stderr)

# --- perceptual narrowing ---
nar = load("narrowing")
if nar is not None:
    import matplotlib.pyplot as plt
    g = nar[(nar.family == "gru")]
    lines = [("developmental", "native", "native, developmental", "#1b9e77", "-"),
             ("developmental", "nonnative", "non-native, developmental", "#d95f02", "-"),
             ("mature", "native", "native, mature front end", "#1b9e77", ":"),
             ("mature", "nonnative", "non-native, mature front end", "#d95f02", ":"),
             ("exposed", "nonnative", "non-native, exposed control", "#7570b3", "--")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2))
    for r, s in enumerate(["vision", "audio"]):
        for c, (met, title) in enumerate([("swap_voe", "identity-swap VoE (d_z)"), ("dprime", "representational d′")]):
            ax = axes[r, c]
            for sch, con, lab, col, ls in lines:
                sub = g[(g.sense == s) & (g.schedule == sch) & (g.contrast == con)].groupby("step")[met]
                m, e = sub.mean(), sub.std()
                x = m.index.values + 1
                ax.plot(x, m.values, ls, color=col, label=lab, lw=1.6)
                ax.fill_between(x, (m - e).values, (m + e).values, color=col, alpha=0.12)
            ax.set_xscale("log"); ax.set_title(f"{s}: {title}", fontsize=10); ax.axhline(0, color="k", lw=0.5)
            if r == 1: ax.set_xlabel("training step + 1 (one continuous run = age)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Perceptual narrowing: only identities {0,1} ever occur; {2,3} is the non-native contrast (GRU, 3 seeds)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{FIG}/narrowing.png", dpi=150)
    t = g[g.sense != "touch"].groupby(["sense", "schedule", "contrast", "step"])[["swap_voe", "dprime"]].mean().round(2)
    open(f"{R}/narrowing_summary.md", "w").write(t.to_markdown())
