"""Multi-panel, journal-style figures for the paper (report/figures/fig*.png|pdf).

Fig. 1  Overview: system schematic, what each sense delivers by age, front-end maturity, regime design.
Fig. 2  Development: trajectories against the information ceiling, own-age probes, fraction of
        ceiling reached, body senses, milestone order.
Fig. 3  Costs and benefits: clean vs newborn-input probes, hearing, effect sizes, compute.
Fig. 4  Fusion and expectation: occluded tracking, fill-in, unimodal VoE, fused VoE by event and
        channel, an example surprise trace.
Fig. 5  Companion studies (critical period, sense-onset crowding, perceptual narrowing).
ED 1-3  Extended Data: model outputs, all-sense trajectories, ceilings and own-age probes.

Reads results/*/seed*/metrics.json, results/extra.json, results/fusion_voe.json and
docs/data/model_outputs.json (scripts/export_site_data.py). Usage: python scripts/make_paper_figures.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from dwm import capability as cap  # noqa: E402
from dwm import senses as S  # noqa: E402
from dwm.env import generate  # noqa: E402

RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "report", "figures")
REGIMES = ["curriculum_only", "grown", "growth_only", "adult_scratch", "shuffled_order", "reversed_immaturity"]
LABEL = {"curriculum_only": "Immature input", "grown": "Immature + growth", "growth_only": "Growth only",
         "adult_scratch": "Adult input", "shuffled_order": "Shuffled quality", "reversed_immaturity": "Reversed"}
SHORT = {"curriculum_only": "Immature", "grown": "Imm.+grow", "growth_only": "Grow only",
         "adult_scratch": "Adult", "shuffled_order": "Shuffled", "reversed_immaturity": "Reversed"}
COL = {"curriculum_only": "#2a78d6", "grown": "#1baf7a", "growth_only": "#eda100", "adult_scratch": "#eb6834",
       "shuffled_order": "#e87ba4", "reversed_immaturity": "#008300"}
SCOL = {"vision": "#2a78d6", "audio": "#eb6834", "touch": "#1baf7a", "proprio": "#eda100", "all": "#222222",
        "fused": "#222222"}
SNAME = {"vision": "Vision", "audio": "Hearing", "touch": "Touch", "proprio": "Body sense"}
GREY, MUTED = "#8a8a8a", "#5d5d63"
STAGES = cap.STAGES
MM = 1 / 25.4

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "DejaVu Sans"], "font.size": 6.5,
    "axes.titlesize": 7, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 5.8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 1.2, "lines.markersize": 3, "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})


# ------------------------------------------------------------------ data
def load_runs():
    runs = {}
    for p in glob.glob(os.path.join(RES, "*", "seed*", "metrics.json")):
        regime, seed = p.split(os.sep)[-3], int(p.split(os.sep)[-2][4:])
        runs[(regime, seed)] = json.load(open(p))
    return runs


RUNS = load_runs()
EXTRA = json.load(open(os.path.join(RES, "extra.json"))) if os.path.exists(os.path.join(RES, "extra.json")) else {}
FV = json.load(open(os.path.join(RES, "fusion_voe.json"))) if os.path.exists(os.path.join(RES, "fusion_voe.json")) else {}
MO_P = os.path.join(ROOT, "docs", "data", "model_outputs.json")
MO = json.load(open(MO_P)) if os.path.exists(MO_P) else {}


def series(regime, sense, key, sub=None):
    """(n_seeds, 4) array of a metric over the four checkpoints."""
    rows = []
    for (rg, _), r in sorted(RUNS.items()):
        t = r["senses"].get(sense, {}).get(regime) if rg == regime else None
        if t:
            rows.append([(x[key][sub] if sub else x[key]) if key in x else np.nan for x in t])
    return np.array(rows, float) if rows else np.full((0, 4), np.nan)


def fusion_series(group, name, metric):
    rows = []
    for (rg, _), r in sorted(RUNS.items()):
        if rg == "grown" and r.get("fusion"):
            rows.append([r["fusion"][str(a)][group][name][metric] for a in STAGES])
    return np.array(rows, float)


def panel(ax, letter, x=-0.22, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def band(ax, x, arr, color, label=None, ls="-", marker="o", lw=1.2, alpha=0.15):
    if len(arr) == 0:
        return
    m = np.nanmean(arr, 0)
    ax.plot(x, m, ls, color=color, marker=marker, label=label, lw=lw, ms=2.8, mec="white", mew=0.4)
    if len(arr) > 1:
        s = np.nanstd(arr, 0)
        ax.fill_between(x, m - s, m + s, color=color, alpha=alpha, lw=0)


def ceiling(sense, metric):
    c = EXTRA.get("ceiling", {}).get(sense)
    if not c:
        return None, None
    ages = sorted(int(a) for a in c)
    return np.array(ages), np.array([c[str(a)][metric] for a in ages])


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def n_of(regime, sense="vision"):
    return len(series(regime, sense, "probe_adult", "pos_r2"))


# ------------------------------------------------------------------ Fig. 1
def box(ax, x, y, w, h, text, fc, ec="none", fs=6, color="black", weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015", fc=fc, ec=ec, lw=0.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, weight=weight)


def arrow(ax, x0, y0, x1, y1, color=MUTED):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, lw=0.6, color=color))


def fig1():
    fig = plt.figure(figsize=(180 * MM, 128 * MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.8, 1.05], width_ratios=[1.25, 1, 1], hspace=0.3, wspace=0.35)
    # a: schematic
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    panel(ax, "a", x=-0.02, y=0.98)
    box(ax, 0.0, 0.33, 0.11, 0.34, "Synthetic\nnursery\n\ntoy · hand ·\nscreen · sound", "#f2f0eb", fs=6)
    ys = [0.74, 0.52, 0.30, 0.08]
    for s, y in zip(["vision", "audio", "touch", "proprio"], ys):
        c = SCOL[s]
        arrow(ax, 0.115, 0.5, 0.165, y + 0.07)
        box(ax, 0.17, y, 0.15, 0.14, f"{SNAME[s]} front end\n" + r"$\tilde o^{s}_t=\Phi_s(o^{s}_t;\,a)$", c + "26", fs=5.6)
        arrow(ax, 0.322, y + 0.07, 0.356, y + 0.07)
        box(ax, 0.358, y, 0.245, 0.14, f"{SNAME[s]} world model\n" + r"$z_t=e(\tilde o_t)$,  $\hat z_{t+1}=z_t+f(z_{t-k+1:t},u_t)$", c + "40", fs=5.4)
        arrow(ax, 0.605, y + 0.07, 0.64, 0.5)
    box(ax, 0.642, 0.32, 0.16, 0.36, "Fusion world model\n\nmasked mean of\nprojected latents,\nwindowed dynamics,\n30% modality dropout", "#22222218", fs=5.6)
    arrow(ax, 0.804, 0.5, 0.835, 0.5)
    box(ax, 0.838, 0.62, 0.16, 0.15, "Linear probes\n(position, identity)", "#f2f0eb", fs=5.6)
    box(ax, 0.838, 0.425, 0.16, 0.15, "Violation of expectation\n(surprise)", "#f2f0eb", fs=5.6)
    box(ax, 0.838, 0.23, 0.16, 0.15, "Occluded tracking,\ncross-modal fill-in", "#f2f0eb", fs=5.6)
    ax.text(0.245, 0.93, "fixed, age-indexed", ha="center", fontsize=5.6, color=MUTED, style="italic")
    ax.text(0.473, 0.93, "learned; one continuous run per sense", ha="center", fontsize=5.6, color=MUTED, style="italic")
    ax.text(0.72, 0.93, "learned on frozen sense latents", ha="center", fontsize=5.6, color=MUTED, style="italic")

    # b: what each sense delivers
    sub = gs[1, 0].subgridspec(3, 4, hspace=0.12, wspace=0.08)
    d = generate(300, seed=777)
    i = next(k for k in range(300) if d["contact"][k].sum() >= 3 and (d["visible"][k] < 0.5).sum() >= 1)
    t = int(np.argmax(d["contact"][i]))
    v = torch.from_numpy(d["vision"][i, t:t + 1]); au = torch.from_numpy(d["audio"][i, :]); to = torch.from_numpy(d["touch"][i, t:t + 1])
    g = torch.Generator().manual_seed(0)
    for j, a in enumerate(STAGES):
        axv = fig.add_subplot(sub[0, j]); axv.imshow(S.vision(v, a)[0].permute(1, 2, 0).numpy(), interpolation="nearest"); axv.set_xticks([]); axv.set_yticks([])
        axv.set_title(f"{a} mo", fontsize=6, loc="center", weight="normal")
        axa = fig.add_subplot(sub[1, j]); spec = S.audio(au, a, g).view(-1, 2, 32)[:, 0].T.numpy()
        axa.imshow(spec, origin="lower", aspect="auto", cmap="Oranges", vmin=0, vmax=0.6, interpolation="nearest"); axa.set_xticks([]); axa.set_yticks([])
        axt = fig.add_subplot(sub[2, j]); axt.imshow(S.touch(to, a, g).view(8, 8).numpy(), cmap="Greens", vmin=0, vmax=1, interpolation="nearest"); axt.set_xticks([]); axt.set_yticks([])
        if j == 0:
            panel(axv, "b", x=-0.75, y=1.15)
            axv.set_ylabel("Vision", fontsize=6); axa.set_ylabel("Hearing\n(L ear, time→)", fontsize=5.5); axt.set_ylabel("Touch", fontsize=6)
        for axx in (axv, axa, axt):
            for sp in axx.spines.values():
                sp.set_visible(True); sp.set_linewidth(0.4); sp.set_color("#bbbbbb")

    # c: front-end maturity
    ax = fig.add_subplot(gs[1, 1]); panel(ax, "c")
    ages = np.linspace(0, 12, 97)
    mat = {
        "vision": [cap.interp(cap.VISION_ACUITY_CPD, a) / 30 for a in ages],
        "audio": [cap.interp(cap.AUDIO_BINAURAL, a) for a in ages],
        "touch": [1 - (cap.interp(cap.TOUCH_BLUR, a) - 0.3) / (1.2 - 0.3) * 0.6 for a in ages],
        "proprio": [1 - (cap.interp(cap.PROPRIO_NOISE, a) - 0.02) / 0.1 * 0.3 for a in ages],
    }
    for s, y in mat.items():
        ax.plot(ages, y, color=SCOL[s], label=SNAME[s])
    ax.set_xlabel("age (months)"); ax.set_ylabel("front-end maturity (a.u.)"); ax.set_xticks(STAGES); ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right"); ax.set_title("Each sense on its own clock", fontsize=6.5)

    # d: regime design
    ax = fig.add_subplot(gs[1, 2]); panel(ax, "d")
    sched = {"curriculum_only": STAGES, "grown": STAGES, "growth_only": [12] * 4, "adult_scratch": [12] * 4,
             "reversed_immaturity": STAGES[::-1]}
    off = {"curriculum_only": -0.3, "grown": 0.3, "growth_only": 0.3, "adult_scratch": -0.3, "reversed_immaturity": 0}
    for r, s in sched.items():
        sizes = [6, 12, 18, 24] if r in ("grown", "growth_only") else [24] * 4
        ax.plot(np.arange(1, 5), np.array(s) + off[r], "-", color=COL[r], lw=0.9)
        ax.scatter(np.arange(1, 5), np.array(s) + off[r], s=sizes, color=COL[r], zorder=3, label=LABEL[r], edgecolor="white", lw=0.3)
    ax.scatter(np.arange(1, 5), [6.5] * 4, marker="s", s=10, color=COL["shuffled_order"], label=LABEL["shuffled_order"] + " (random age per batch)")
    ax.set_xticks([1, 2, 3, 4]); ax.set_xticklabels(["stage 1", "2", "3", "4"]); ax.set_yticks(STAGES)
    ax.set_ylabel("input quality (age, months)"); ax.set_ylim(-1.2, 15)
    ax.legend(loc="upper center", ncol=2, fontsize=4.8, bbox_to_anchor=(0.45, -0.17), handletextpad=0.2, columnspacing=0.6)
    ax.text(0.85, 13.7, "marker size = network width", fontsize=5, color=MUTED)
    ax.set_title("Six training regimes", fontsize=6.5)
    save(fig, "fig1_overview")


# ------------------------------------------------------------------ Fig. 2
def fig2():
    fig, axs = plt.subplots(2, 3, figsize=(180 * MM, 108 * MM), gridspec_kw={"hspace": 0.55, "wspace": 0.38})
    # a vision trajectories + ceiling
    for ax, sense, letter, title in [(axs[0, 0], "vision", "a", "Vision: toy position"), (axs[0, 2], "audio", "c", "Hearing: toy position")]:
        panel(ax, letter)
        ca, cv = ceiling(sense, "pos_r2")
        if ca is not None:
            ax.plot(ca, cv, "--", color=GREY, lw=1, label="information ceiling")
        for r in REGIMES:
            band(ax, STAGES, series(r, sense, "probe_adult", "pos_r2"), COL[r], LABEL[r])
        ax.set_xticks(STAGES); ax.set_xlabel("checkpoint (months)"); ax.set_ylabel("probe R² (adult-quality test input)")
        ax.set_title(title); ax.set_ylim(-0.05, 1.0)
    axs[0, 0].legend(loc="lower right", fontsize=5, handlelength=1.4)
    # b own-age
    ax = axs[0, 1]; panel(ax, "b")
    own = EXTRA.get("own_age", {})
    for r in REGIMES:
        o = own.get(r, {}).get("vision")
        if o:
            arr = np.array([[d["pos_r2"] for d in o.get(str(a), [])] for a in STAGES], dtype=object)
            n = min(len(x) for x in arr)
            if n:
                band(ax, STAGES, np.array([[arr[k][j] for k in range(4)] for j in range(n)], float), COL[r], LABEL[r])
    ca, cv = ceiling("vision", "pos_r2")
    if ca is not None:
        ax.plot(ca, cv, "--", color=GREY, lw=1)
    ax.set_xticks(STAGES); ax.set_xlabel("checkpoint (months)"); ax.set_ylabel("probe R² (own-age input)")
    ax.set_title("Vision, probed at its own age"); ax.set_ylim(-0.05, 1.0)
    if not own:
        ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center", color=GREY)
    # d fraction of ceiling reached (age-matched input)
    ax = axs[1, 0]; panel(ax, "d")
    for sense, ls in [("vision", "-"), ("audio", "--")]:
        ca, cv = ceiling(sense, "pos_r2")
        if ca is None:
            continue
        cmap = dict(zip(ca.tolist(), cv.tolist()))
        for r, ages in [("curriculum_only", STAGES), ("adult_scratch", [12] * 4)]:
            if r == "curriculum_only" and own.get(r, {}).get(sense):
                o = own[r][sense]
                arr = np.array([[d["pos_r2"] for d in o[str(a)]] for a in STAGES]).T
            else:
                arr = series(r, sense, "probe_adult", "pos_r2")
            frac = arr / np.array([cmap[a] for a in ages])[None]
            band(ax, STAGES, frac, COL[r], f"{SNAME[sense]}, {SHORT[r].lower()}", ls=ls)
    ax.axhline(1, color=GREY, lw=0.6, ls=":")
    ax.set_xticks(STAGES); ax.set_xlabel("checkpoint (months)"); ax.set_ylabel("fraction of ceiling reached")
    ax.set_title("Learning lags what the input allows"); ax.legend(loc="lower right", fontsize=5)
    # e body senses
    ax = axs[1, 1]; panel(ax, "e")
    for r in ["curriculum_only", "adult_scratch", "grown"]:
        band(ax, STAGES, series(r, "proprio", "probe_adult", "pos_r2"), COL[r], marker="s")
        band(ax, STAGES, series(r, "touch", "probe_adult", "type_acc"), COL[r], marker="^", ls="--")
    ax.axhline(0.25, color=GREY, lw=0.6, ls=":"); ax.text(12, 0.27, "chance (touch)", ha="right", fontsize=5, color=GREY)
    ax.plot([], [], "k-s", ms=2.5, label="body sense: hand position R²"); ax.plot([], [], "k--^", ms=2.5, label="touch: toy identity accuracy")
    ax.set_xticks(STAGES); ax.set_xlabel("checkpoint (months)"); ax.set_ylim(0, 1.05); ax.set_title("Body senses start near mature")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.72), fontsize=5)
    # f milestones
    ax = axs[1, 2]; panel(ax, "f")
    ms = EXTRA.get("milestones")
    if ms:
        senses = ["touch", "audio", "vision"]
        ages = [ms["half_ceiling_age_months"][s] or 0 for s in senses]
        ax.bar(range(3), ages, color=[SCOL[s] for s in senses], width=0.6)
        for k, a in enumerate(ages):
            ax.text(k, a + 0.15, f"{a}", ha="center", fontsize=6)
        ax.set_xticks(range(3)); ax.set_xticklabels([SNAME[s] for s in senses])
        ax.set_ylabel("age at half of 12-mo ceiling (months)")
        ax.set_title(f"Half-ceiling age (Kendall τ = {ms['kendall_tau_b_approx']:.2f})")
        ax.text(0.02, 0.97, "infant order: touch < hearing < vision\n(set by the front-end tables)", transform=ax.transAxes, va="top", fontsize=5, color=MUTED)
    save(fig, "fig2_development")


# ------------------------------------------------------------------ Fig. 3
def hedges_ci(g, na, nb):
    se = np.sqrt((na + nb) / (na * nb) + g ** 2 / (2 * (na + nb)))
    return g - 1.96 * se, g + 1.96 * se


def strip(ax, regimes, sense, key, sub, ylabel, title, hatch_key=None):
    for k, r in enumerate(regimes):
        a = series(r, sense, key, sub)[:, -1]
        if len(a) == 0:
            continue
        ax.bar(k, a.mean(), color=COL[r], width=0.62, alpha=0.9)
        ax.scatter(k + np.linspace(-0.12, 0.12, len(a)), a, s=5, color="black", zorder=3, lw=0)
        if len(a) > 1:
            ax.errorbar(k, a.mean(), yerr=a.std(ddof=1), color="black", lw=0.7, capsize=1.5)
    ax.set_xticks(range(len(regimes))); ax.set_xticklabels([SHORT[r] for r in regimes], rotation=35, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title); ax.axhline(0, color="black", lw=0.5)


def fig3():
    fig, axs = plt.subplots(1, 4, figsize=(180 * MM, 54 * MM), gridspec_kw={"wspace": 1.0, "width_ratios": [1.05, 1.05, 1.0, 1]})
    panel(axs[0], "a"); strip(axs[0], REGIMES, "vision", "probe_newborn", "pos_r2", "probe R² on newborn-quality input", "Vision, newborn input")
    panel(axs[1], "b"); strip(axs[1], REGIMES, "audio", "probe_adult", "pos_r2", "probe R² (clean input)", "Hearing: localisation")
    axs[1].set_ylim(0, 0.6)
    ax = axs[2]; panel(ax, "c")
    comps = EXTRA.get("stats", {}).get("comparisons", {})
    names = list(comps)
    for k, nme in enumerate(names):
        c = comps[nme]; g = c["hedges_g"]; lo, hi = hedges_ci(g, len(c["a"]), len(c["b"]))
        ax.plot([lo, hi], [k, k], color=MUTED, lw=0.8); ax.plot(g, k, "o", color="black", ms=3)
        ax.text(max(hi, 0) + 0.3, k, f"p={c['p']:.2f}, n={len(c['a'])}/{len(c['b'])}", va="center", fontsize=4.8, color=MUTED)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_yticks(range(len(names))); ax.set_yticklabels([n.replace(": ", ":\n", 1) for n in names], fontsize=4.8)
    ax.invert_yaxis(); ax.set_xlabel("Hedges' g (95% CI)"); ax.set_title("Effect sizes at 12 months")
    ax.set_xlim(-6, 9)
    ax = axs[3]; panel(ax, "d")
    comp = EXTRA.get("compute", {}).get("vision")
    if comp:
        rel = {"grown": comp["ratio_grown_to_fixed"], "growth_only": comp["ratio_grown_to_fixed"]}
        for r in REGIMES:
            a = series(r, "vision", "probe_adult", "pos_r2")[:, -1]
            if len(a):
                ax.errorbar(rel.get(r, 1.0) + (0.02 if r in ("adult_scratch", "growth_only", "reversed_immaturity") else -0.02) * (REGIMES.index(r) % 3),
                            a.mean(), yerr=a.std(ddof=1) if len(a) > 1 else 0, fmt="o", color=COL[r], ms=3.5, capsize=1.5, lw=0.7, label=SHORT[r])
        ax.set_xlim(0.3, 1.15); ax.set_xlabel("training compute\n(parameter-updates, rel. to fixed)"); ax.set_ylabel("vision probe R² at 12 mo")
        ax.set_title("Growth halves compute"); ax.legend(fontsize=4.8, loc="lower right", handletextpad=0.1)
    save(fig, "fig3_costs_benefits")


# ------------------------------------------------------------------ Fig. 4
def fig4():
    fig = plt.figure(figsize=(180 * MM, 104 * MM))
    gs = fig.add_gridspec(2, 3, hspace=0.6, wspace=0.42)
    ax = fig.add_subplot(gs[0, 0]); panel(ax, "a")
    for name, lab, col in [("fused", "fused world model", SCOL["fused"]), ("vision_memory", "vision memory", SCOL["vision"]),
                           ("audio_only", "hearing alone", SCOL["audio"])]:
        band(ax, STAGES, fusion_series("occluded_tracking", name, "x_r2"), col, lab)
    ax.set_xticks(STAGES); ax.set_xlabel("checkpoint (months)"); ax.set_ylabel("hidden-toy x position R²")
    ax.set_title("Tracking a fully hidden toy"); ax.legend(loc="upper left")
    ax = fig.add_subplot(gs[0, 1]); panel(ax, "b")
    items = [("vision_only", "vision\nalone", SCOL["vision"]), ("fused_all", "fused,\nall", SCOL["fused"]),
             ("fused_no_vision", "fused,\nno vision", "#7a7a7a"), ("audio_only", "hearing\nalone", SCOL["audio"])]
    for k, (n, lab, col) in enumerate(items):
        a = fusion_series("fill_in", n, "pos_r2")[:, -1]
        ax.bar(k, a.mean(), color=col, width=0.62); ax.scatter(k + np.linspace(-0.1, 0.1, len(a)), a, s=5, color="white", zorder=3, edgecolor="black", lw=0.3)
        if len(a) > 1:
            ax.errorbar(k, a.mean(), yerr=a.std(ddof=1), color="black", lw=0.7, capsize=1.5)
    ax.set_xticks(range(4)); ax.set_xticklabels([x[1] for x in items]); ax.set_ylabel("visible-toy position R² (12 mo)")
    ax.set_title("Cross-modal fill-in")
    ax = fig.add_subplot(gs[0, 2]); panel(ax, "c")
    for r in REGIMES:
        band(ax, STAGES, series(r, "vision", "voe", "voe_index"), COL[r], SHORT[r])
    ax.axhline(0, color="black", lw=0.5); ax.set_xticks(STAGES); ax.set_xlabel("checkpoint (months)")
    ax.set_ylabel("VoE index (vision model)"); ax.set_title("No vision model expects the toy back"); ax.set_ylim(-0.4, 0.4)
    ax.legend(ncol=2, fontsize=4.8, loc="upper left")
    # d fused VoE by event x channel at 12 mo
    ax = fig.add_subplot(gs[1, 0:2]); panel(ax, "d", x=-0.1)
    events = [("vanish_all", "toy vanishes\nfrom every sense"), ("vanish_visual", "vanishes from sight,\nkeeps sounding"), ("silent_toy", "silent toy\nvanishes")]
    chans = ["vision", "audio", "touch", "proprio", "all"]
    cname = {"vision": "vision channel", "audio": "hearing channel", "touch": "touch channel", "proprio": "body channel", "all": "all channels"}
    seeds = list(FV)
    ok = seeds and "vanish_all" in FV[seeds[0]].get("12", {})
    for ai, age in enumerate(["0", "12"]):
        for e, (ev, elab) in enumerate(events):
            for c, ch in enumerate(chans):
                vals = [FV[s][age][ev][ch]["voe_index"] for s in seeds if ok and ev in FV[s].get(age, {})]
                if not vals:
                    continue
                x = e * 2.6 + ai * 1.2 + c * 0.2
                ax.bar(x, np.mean(vals), width=0.18, color=SCOL[ch], alpha=1 if age == "12" else 0.45,
                       label=cname[ch] if (e == 0 and age == "12") else None)
                if len(vals) > 1:
                    ax.errorbar(x, np.mean(vals), yerr=np.std(vals, ddof=1), color="black", lw=0.5, capsize=1)
            ax.text(e * 2.6 + ai * 1.2 + 0.4, -0.52, f"{age} mo", ha="center", fontsize=5.2, color=MUTED)
        pass
    for e, (ev, elab) in enumerate(events):
        ax.text(e * 2.6 + 1.0, 0.95, elab, ha="center", fontsize=5.8, va="top")
    ax.axhline(0, color="black", lw=0.5); ax.set_xticks([]); ax.set_ylim(-0.55, 1.0); ax.set_xlim(-0.3, 7.6)
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("VoE index (fused model)"); ax.set_title("Which sense notices? Dissociating events (n = %d seeds)" % len(seeds))
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.32), fontsize=5.2)
    if not ok:
        ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center", color=GREY)
    # e example trace
    ax = fig.add_subplot(gs[1, 2]); panel(ax, "e")
    m = MO.get("curriculum_only", {}).get("12")
    if m:
        ep = json.load(open(os.path.join(ROOT, "docs", "data", "episodes.json")))["episodes"]["voe_possible"]
        hid = [k for k, v in enumerate(ep["visible"]) if v < 0.5]
        if hid:
            ax.axvspan(hid[0] - 0.5, hid[-1] + 0.5, color="#eeeeee", lw=0)
            ax.text((hid[0] + hid[-1]) / 2, max(m["possible"]) * 1.02, "hidden", ha="center", fontsize=5, color=MUTED)
        ax.plot(range(1, 20), m["possible"][1:], "-o", color=SCOL["vision"], ms=2, label="possible (reappears)")
        ax.plot(range(1, 20), m["impossible"][1:], "-o", color=COL["adult_scratch"], ms=2, label="impossible (vanished)")
        ax.set_xlabel("frame"); ax.set_ylabel("surprise (prediction MSE)"); ax.set_title("A vision model is surprised by reappearance")
        ax.legend(fontsize=5, loc="upper right")
    save(fig, "fig4_fusion_expectation")


# ------------------------------------------------------------------ Fig. 5 (companion studies)
def fig5():
    src = [("e2b_window.png", "a", "Critical period: a 3-month blur window\nhurts only when it comes first"),
           ("onset_order.png", "b", "Sense-onset order under joint training: in the biological order\nthe late-arriving vision encoder is worst"),
           ("narrowing.png", "c", "Perceptual narrowing emerges without being built in")]
    fig = plt.figure(figsize=(180 * MM, 170 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.75, 1.25], width_ratios=[0.8, 1.3], hspace=0.12, wspace=0.05)
    for spec, (f, l, t) in zip([gs[0, 0], gs[0, 1], gs[1, :]], src):
        ax = fig.add_subplot(spec)
        p = os.path.join(OUT, f)
        if os.path.exists(p):
            ax.imshow(plt.imread(p))
        ax.axis("off"); panel(ax, l, x=-0.01, y=1.01); ax.set_title(t, fontsize=6.5, loc="left", x=0.05, weight="normal")
    save(fig, "fig5_companion")


# ------------------------------------------------------------------ Extended Data
def ed1():
    from PIL import Image
    rows = [("curriculum_only", 0), ("curriculum_only", 12), ("adult_scratch", 0), ("adult_scratch", 12), ("grown", 3), ("reversed_immaturity", 12)]
    fig, axs = plt.subplots(len(rows), 1, figsize=(180 * MM, 150 * MM), gridspec_kw={"hspace": 0.45})
    for ax, (r, a) in zip(axs, rows):
        p = os.path.join(ROOT, "docs", "data", "models", f"{r}_{a}.png")
        if os.path.exists(p):
            im = np.asarray(Image.open(p))[:, : 33 * 14 - 1]
            ax.imshow(im, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([16, 49, 82]); ax.set_yticklabels(["given", "rebuilt", "predicted"], fontsize=5.5)
        for s in ax.spines.values():
            s.set_visible(False)
        info = MO.get(r, {}).get(str(a), {})
        ax.set_title(f"{LABEL[r]}, {a}-month checkpoint (trained on {info.get('input_age', '?')}-month input)", fontsize=6.2, weight="normal")
    save(fig, "ed1_model_outputs")


def ed2():
    fig, axs = plt.subplots(2, 4, figsize=(180 * MM, 82 * MM), gridspec_kw={"hspace": 0.6, "wspace": 0.4})
    for j, sense in enumerate(["vision", "audio", "touch", "proprio"]):
        for i, (key, sub, lab) in enumerate([("probe_adult", "pos_r2", "position R²"), ("probe_adult", "type_acc", "identity accuracy")]):
            ax = axs[i, j]
            for r in REGIMES:
                band(ax, STAGES, series(r, sense, key, sub), COL[r], LABEL[r])
            ax.set_xticks(STAGES); ax.set_title(f"{SNAME[sense]}: {lab}", fontsize=6); ax.set_xlabel("checkpoint (months)")
            if sub == "type_acc":
                ax.axhline(0.25, color=GREY, lw=0.5, ls=":")
            if i == 0 and j == 0:
                panel(ax, "a")
    axs[0, 0].legend(fontsize=4.6, loc="lower right")
    save(fig, "ed2_all_senses")


def ed3():
    fig, axs = plt.subplots(1, 3, figsize=(180 * MM, 52 * MM), gridspec_kw={"wspace": 0.4})
    c = EXTRA.get("ceiling", {})
    for sense in c:
        ages = sorted(int(a) for a in c[sense])
        axs[0].plot(ages, [c[sense][str(a)]["pos_r2"] for a in ages], "-o", color=SCOL[sense], label=SNAME[sense], ms=2.5)
        axs[1].plot(ages, [c[sense][str(a)]["type_acc"] for a in ages], "-o", color=SCOL[sense], label=SNAME[sense], ms=2.5)
    for ax, t, l in [(axs[0], "Ceiling: position R²", "a"), (axs[1], "Ceiling: identity accuracy", "b")]:
        ax.set_xticks(STAGES); ax.set_xlabel("age of input (months)"); ax.set_title(t); panel(ax, l); ax.legend(fontsize=5)
    axs[1].axhline(0.25, color=GREY, lw=0.5, ls=":")
    ax = axs[2]; panel(ax, "c")
    own = EXTRA.get("own_age", {})
    for r in REGIMES:
        o = own.get(r, {}).get("audio")
        if o:
            arr = np.array([[d["pos_r2"] for d in o[str(a)]] for a in STAGES]).T
            band(ax, STAGES, arr, COL[r], SHORT[r])
    ax.set_xticks(STAGES); ax.set_xlabel("checkpoint (months)"); ax.set_title("Hearing, probed at its own age"); ax.legend(fontsize=4.8)
    save(fig, "ed3_ceiling_ownage")


if __name__ == "__main__":
    names = sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4", "fig5", "ed1", "ed2", "ed3"]
    for n in names:
        globals()[n]()
