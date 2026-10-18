"""Extended Data Fig. S4 of paper 1 from results/future_work/summary.json (scripts/analyze_future_work.py)."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
S = json.load(open(os.path.join(ROOT, "results", "future_work", "summary.json")))
OUT = os.path.join(ROOT, "papers", "world-models", "report", "figures", "ed4_future_work.png")
BLUE, LIGHT, ORANGE, GREEN, PURPLE = "#2a78d6", "#7fb2ec", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 2})


def line(ax, d, color, label, ls="-", marker="o"):
    x = [int(k) for k in d]
    m, s = np.array([d[k]["mean"] for k in d]), np.array([d[k]["sd"] for k in d])
    ax.errorbar(x, m, yerr=s, color=color, ls=ls, marker=marker, ms=4, capsize=2, label=label)


fig, ax = plt.subplots(2, 3, figsize=(7.4, 5.0))
for a, (sense, title) in zip(ax[0, :2], [("vision", "a  Vision"), ("audio", "b  Hearing")]):
    line(a, S["staged_" + sense], BLUE, "staged")
    line(a, S["matched_" + sense], ORANGE, "matched", ls="--", marker="s")
    a.set_xticks([0, 3, 6, 12]); a.set_xlabel("age (months)"); a.set_ylabel("toy position, $R^2$"); a.set_title(title)
ax[0, 0].legend(frameon=False, fontsize=6.5, loc="lower right")
ages = ["0", "3", "6", "12"]
for a, key, ylab, title in [(ax[0, 2], "top10", "Jacobian energy, top 10 modes", "c  Input modes used"),
                            (ax[1, 0], "eff_rank", "effective rank of Jacobian", "d  Breadth of response")]:
    for regime, col, lab in [("curriculum_only", BLUE, "immature input"), ("adult_scratch", ORANGE, "adult input")]:
        line(a, {k: S["spectral"][regime][k][key] for k in ages}, col, lab)
    a.set_xticks([0, 3, 6, 12]); a.set_xlabel("checkpoint (months)"); a.set_ylabel(ylab); a.set_title(title)
ax[0, 2].axhline(S["spectral"]["input"]["var_top10"], color=INK2, ls=":", lw=1)
ax[0, 2].text(12, S["spectral"]["input"]["var_top10"] + 0.02, "input variance in top 10", ha="right", fontsize=6.5, color=INK2)
ax[0, 2].legend(frameon=False, fontsize=6.5)
names = [("voe_pixel_reference", "pixel (main)", BLUE), ("voe_gauss.voe_nll", "Gaussian\nNLL", GREEN),
         ("voe_jepa_vr.voe_latent", "latent\n(VICReg)", PURPLE)]
vals = [S["voe_pixel_reference"]["12"], S["voe_gauss"]["voe_nll"]["12"], S["voe_jepa_vr"]["voe_latent"]["12"]]
ax[1, 1].axhline(0, color=INK2, lw=0.8)
for i, (v, (_, lab, col)) in enumerate(zip(vals, names)):
    ax[1, 1].bar(i, v["mean"], yerr=v["sd"], color=col, alpha=0.85, capsize=3, width=0.6)
    ax[1, 1].scatter(np.full(len(v["values"]), i) + np.linspace(-0.12, 0.12, len(v["values"])), v["values"], color=INK, s=8, zorder=3)
ax[1, 1].set_xticks(range(3)); ax[1, 1].set_xticklabels([n[1] for n in names], fontsize=7)
ax[1, 1].set_ylabel("VoE index (12 months)"); ax[1, 1].set_title("e  Surprise at reappearance")
probe = [S["staged_vision"]["12"], S["voe_gauss"]["probe_pos_r2"]["12"], S["voe_jepa_vr"]["probe_pos_r2"]["12"]]
for i, (v, (_, lab, col)) in enumerate(zip(probe, names)):
    ax[1, 2].bar(i, v["mean"], yerr=v["sd"], color=col, alpha=0.85, capsize=3, width=0.6)
    ax[1, 2].scatter(np.full(len(v["values"]), i) + np.linspace(-0.12, 0.12, len(v["values"])), v["values"], color=INK, s=8, zorder=3)
ax[1, 2].set_xticks(range(3)); ax[1, 2].set_xticklabels([n[1] for n in names], fontsize=7)
ax[1, 2].set_ylabel("position $R^2$ (12 months)"); ax[1, 2].set_title("f  Tracking the toy")
fig.tight_layout()
fig.savefig(OUT, dpi=200)
print("wrote", OUT)
