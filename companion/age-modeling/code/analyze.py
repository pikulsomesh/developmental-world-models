"""Summaries + figures for E1/E2/E3."""
import json, os, sys, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "figs"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
COL = {"fixed": "#3b6fb6", "grow": "#d08a1f", "prune": "#2f9e6e", "hiercp": "#8e5bb5", "separate": "#c0392b"}
LAB = {"fixed": "Snapshot (one run, fixed size)", "grow": "Grow channels", "prune": "Overproduce then prune",
       "hiercp": "Snapshot + bottom-up plasticity closing", "separate": "Separate model per age"}
lines = []


def ms(v):
    v = np.asarray(v)
    return f"{v.mean():.2f} ± {v.std():.2f}"


def load(n):
    f = f"results_{n}.json"
    return json.load(open(f)) if os.path.exists(f) else []


# ---------- E1 ----------
E1 = load("E1")
if E1:
    metrics = ["congruent", "shape_only", "texture_only", "shape_bias", "blur_robust", "params"]
    lines.append("## E1 final (month 12), mean ± sd over seeds\n")
    lines.append("| strategy | input | " + " | ".join(metrics) + " |")
    lines.append("|" + "---|" * (len(metrics) + 2))
    groups = collections.defaultdict(list)
    for r in E1:
        c = r["cfg"]
        if c["strategy"] == "separate" and c["age"] != 12:
            continue
        groups[(c["strategy"], c["input"])].append(r["traj"][-1])
    for (s, i), rs in sorted(groups.items()):
        lines.append(f"| {s} | {i} | " + " | ".join(ms([x[m] for x in rs]) if m != "params" else str(int(np.mean([x[m] for x in rs]))) for m in metrics) + " |")
    # trajectories
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharex=True)
    for row, inp in enumerate(["dev", "clear"]):
        for col, m in enumerate(["congruent", "shape_only", "texture_only", "shape_bias"]):
            ax = axes[row, col]
            for s in ["fixed", "grow", "prune", "hiercp"]:
                T = [r["traj"] for r in E1 if r["cfg"]["strategy"] == s and r["cfg"]["input"] == inp]
                if not T:
                    continue
                A = np.array([[t[m] for t in tr] for tr in T])
                x = [t["month"] for t in T[0]]
                ax.plot(x, A.mean(0), color=COL[s], label=LAB[s], lw=1.6)
                ax.fill_between(x, A.mean(0) - A.std(0), A.mean(0) + A.std(0), color=COL[s], alpha=0.15)
            sep = collections.defaultdict(list)
            for r in E1:
                if r["cfg"]["strategy"] == "separate" and r["cfg"]["input"] == inp:
                    sep[r["cfg"]["age"]].append(r["traj"][-1][m])
            if sep:
                ks = sorted(sep)
                ax.errorbar(ks, [np.mean(sep[k]) for k in ks], [np.std(sep[k]) for k in ks], color=COL["separate"],
                            marker="o", ls="--", lw=1.2, ms=4, label=LAB["separate"])
            ax.set_title(f"{m.replace('_', ' ')} ({'acuity schedule' if inp == 'dev' else 'clear input'})")
            if row == 1:
                ax.set_xlabel("simulated age (months)")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/e1_trajectories.png", dpi=150)
    # params over time
    fig, ax = plt.subplots(figsize=(4.6, 3))
    for s in ["fixed", "grow", "prune"]:
        T = [r["traj"] for r in E1 if r["cfg"]["strategy"] == s and r["cfg"]["input"] == "dev"]
        if T:
            ax.plot([t["month"] for t in T[0]], [t["params"] for t in T[0]], color=COL[s], label=LAB[s])
    ax.set_xlabel("simulated age (months)"); ax.set_ylabel("live connections (weights)")
    ax.legend(fontsize=7, frameon=False); fig.tight_layout(); fig.savefig(f"{OUT}/e1_capacity.png", dpi=150)

# ---------- E2 ----------
E2 = load("E2")
if E2:
    lines.append("\n## E2 early deprivation (blur sigma=3 for k months, then 8 months clear)\n")
    lines.append("| strategy | deficit months | congruent | texture_only (fine detail) | shape_only | shape_bias |")
    lines.append("|---|---|---|---|---|---|")
    g = collections.defaultdict(list)
    for r in E2:
        g[(r["cfg"]["strategy"], r["cfg"]["deficit"])].append(r["traj"][-1])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for s in ["fixed", "prune"]:
        ks = sorted(k for (ss, k) in g if ss == s)
        for k in ks:
            rs = g[(s, k)]
            lines.append(f"| {s} | {k} | {ms([x['congruent'] for x in rs])} | {ms([x['texture_only'] for x in rs])} | {ms([x['shape_only'] for x in rs])} | {ms([x['shape_bias'] for x in rs])} |")
        for ax, m in zip(axes, ["texture_only", "shape_bias"]):
            ax.errorbar(ks, [np.mean([x[m] for x in g[(s, k)]]) for k in ks],
                        [np.std([x[m] for x in g[(s, k)]]) for k in ks], marker="o", color=COL[s], label=LAB[s])
            ax.set_xlabel("months of early blur before normal vision"); ax.set_title(m.replace("_", " "))
    axes[0].legend(fontsize=7, frameon=False); fig.tight_layout(); fig.savefig(f"{OUT}/e2_deprivation.png", dpi=150)

# ---------- E3 ----------
E3 = load("E3")
if E3:
    lines.append("\n## E3 sensory onset order (month 12)\n")
    lines.append("| schedule | fused | vision alone (read-out) | audio alone (read-out) | vision probe | audio probe |")
    lines.append("|---|---|---|---|---|---|")
    g = collections.defaultdict(list)
    for r in E3:
        g[r["cfg"]["schedule"]].append(r["final"])
    order = ["simultaneous", "audio_first", "vision_first", "separate_then_fuse", "vision_only", "audio_only"]
    for s in order:
        if s in g:
            rs = g[s]
            lines.append(f"| {s} | " + " | ".join(ms([x[m] for x in rs]) for m in
                         ["fused", "vision_alone", "audio_alone", "probe_vision", "probe_audio"]) + " |")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
    cols = {"simultaneous": "#3b6fb6", "audio_first": "#2f9e6e", "vision_first": "#d08a1f"}
    for ax, m in zip(axes, ["fused", "vision_alone", "audio_alone"]):
        for s, c in cols.items():
            T = [r["traj"] for r in E3 if r["cfg"]["schedule"] == s]
            if T:
                A = np.array([[t[m] for t in tr] for tr in T])
                x = [t["month"] for t in T[0]]
                ax.plot(x, A.mean(0), color=c, label=s.replace("_", " "))
                ax.fill_between(x, A.mean(0) - A.std(0), A.mean(0) + A.std(0), color=c, alpha=0.15)
        ax.set_title(m.replace("_", " ")); ax.set_xlabel("simulated age (months)")
    axes[0].legend(fontsize=7, frameon=False); fig.tight_layout(); fig.savefig(f"{OUT}/e3_onset.png", dpi=150)

E3b = load("E3b")
if E3 and E3b:
    g["audio_first (+3 prenatal, 12 postnatal)"] = [r["final"] for r in E3b]
    lab = {"vision_only": "vision only", "audio_only": "audio only", "simultaneous": "simultaneous",
           "audio_first": "audio first\n(9 mo vision)", "audio_first (+3 prenatal, 12 postnatal)": "audio first\n(12 mo vision)",
           "vision_first": "vision first", "separate_then_fuse": "separate,\nthen fuse"}
    order = ["vision_only", "audio_only", "simultaneous", "audio_first", "audio_first (+3 prenatal, 12 postnatal)", "vision_first", "separate_then_fuse"]
    rs = [g[k] for k in order]
    lines.append("| audio_first, matched vision time (15 mo) | " + " | ".join(ms([x[m] for x in g[order[4]]]) for m in
                 ["fused", "vision_alone", "audio_alone", "probe_vision", "probe_audio"]) + " |")
    fig, ax = plt.subplots(figsize=(8.5, 3))
    x = np.arange(len(order)); w = 0.27
    for j, (m, c, nm) in enumerate([("probe_vision", "#3b6fb6", "vision encoder (linear probe)"),
                                    ("probe_audio", "#2f9e6e", "audio encoder (linear probe)"),
                                    ("fused", "#7f7f7f", "fused read-out")]):
        ax.bar(x + (j - 1) * w, [np.mean([r[m] for r in g[k]]) for k in order], w,
               yerr=[np.std([r[m] for r in g[k]]) for k in order], color=c, label=nm, capsize=2)
    ax.set_xticks(x); ax.set_xticklabels([lab[k] for k in order], fontsize=7.5)
    ax.set_ylim(0, 1); ax.set_ylabel("accuracy at 12 months"); ax.axhline(1 / 8, color="k", lw=0.6, ls=":")
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left"); fig.tight_layout(); fig.savefig(f"{OUT}/e3_encoders.png", dpi=150)

open(f"{OUT}/summary.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
