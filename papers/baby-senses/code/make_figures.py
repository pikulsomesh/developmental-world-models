"""Figures for paper 2. Age tables are copied from the Baby Senses Lab (docs/app/lab-data.js);
model results are read from results/extra.json of the developmental-world-models repository."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "..", "figures") + "/"
C = {"vision": "#2a78d6", "hearing": "#eb6834", "touch": "#1baf7a", "chem": "#6250d6"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 2})

def interp(table, a):
    ks = sorted(table); 
    return np.interp(a, ks, [table[k] for k in ks])

# ---- tables from lab-data.js
ACU_BRAIN = {0: 1.0, 1: 1.5, 2: 2.0, 3: 4.0, 4: 6.0, 5: 10.0, 6: 15.0, 9: 25.0, 12: 30.0, 24: 30.0}
ACU_LOOK = {0: 0.8, 1: 1.0, 2: 1.8, 3: 2.6, 4: 3.4, 6: 4.8, 9: 6.0, 12: 7.5, 18: 9.0, 24: 10.5}
CS = {0: 2, 1: 3, 2: 5, 3: 10, 4: 20, 5: 30, 6: 50, 9: 80, 12: 100, 24: 100}
SAT = {0: 0.1, 1: 0.2, 2: 0.4, 3: 0.7, 4: 0.9, 5: 1.0, 24: 1.0}
H_DB = {0: 25, 3: 20, 6: 12, 12: 6, 24: 3}
H_STEREO = {0: 0.3, 3: 0.5, 6: 0.8, 12: 1.0, 24: 1.0}
H_TREBLE = {0: 12, 1: 6, 3: 4, 6: 1, 12: 0, 24: 0}
a = np.linspace(0, 24, 241)

# ================= Figure 1: milestone timeline per sense
MS = [  # (row, month, label, placement): u/d = above/below, L = left-aligned from the dot
    ("Touch", 0, "responds everywhere;\nrooting, grasp", "uL"), ("Touch", 3, "explores with\nhands and mouth", "d"),
    ("Touch", 18, "walks alone: touch and\nbody sense guide balance", "u"),
    ("Taste and smell", 0, "prefers odours met\nin the womb", "uL"),
    ("Hearing", 0, "prefers mother's\nvoice", "uL"), ("Hearing", 4.5, "recognises\nown name", "d"), ("Hearing", 6, "vowels tune to\nhome language", "uL"),
    ("Hearing", 11, "non-native consonants\nharder", "d"), ("Hearing", 18, "follows spoken\none-step directions", "u"), ("Hearing", 24, "understands\nhundreds of words", "uR"),
    ("Vision", 0, "prefers face-like\npatterns", "uL"), ("Vision", 2.5, "smooth pursuit\ndevelops", "d"), ("Vision", 4, "binocular\ndepth", "uL"),
    ("Vision", 9, "face narrowing to\nfamiliar faces", "d"),
]
SPANS = [  # (row, from, to, label): gradual refinement over a period
    ("Touch", 3, 24, "locating touch in the space around the body improves gradually"),
    ("Hearing", 6, 24, "localising sounds and listening in noise keep improving"),
    ("Vision", 12, 24, "acuity keeps sharpening; adult values only in the preschool years"),
]
rows = ["Vision", "Hearing", "Taste and smell", "Touch"]
col = {"Vision": C["vision"], "Hearing": C["hearing"], "Taste and smell": C["chem"], "Touch": C["touch"]}
SP = 1.7
fig, ax = plt.subplots(figsize=(7.2, 4.9))
ax.grid(axis="y", visible=False)
ax.axvspan(0, 12, color="#f2f1ed", zorder=0, lw=0)
for i, r in enumerate(rows):
    y = i * SP
    ax.plot([-1.5, 24], [y, y], color=GRID, lw=6, solid_capstyle="round", zorder=1)
    for rr, a0, a1, lab in SPANS:
        if rr != r: continue
        ax.plot([a0, a1], [y, y], color=col[r], lw=6, alpha=0.35, solid_capstyle="round", zorder=2)
        ax.text(a1, y - 0.36, lab, ha="right", va="top", fontsize=6.3, color=INK2, style="italic")
    for rr, m, lab, pl in MS:
        if rr != r: continue
        up = pl.startswith("u"); left = pl.endswith("L"); right = pl.endswith("R")
        ax.scatter([m], [y], s=46, color=col[r], edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(lab, (m, y), xytext=(-3 if left else (3 if right else 0), 9 if up else -9), textcoords="offset points",
                    ha="left" if left else ("right" if right else "center"), va="bottom" if up else "top", fontsize=6.8, color=INK, linespacing=1.0)
ax.text(24, 2 * SP - 0.36, "no later milestone in the sources we could verify", ha="right", va="top", fontsize=6.3, color=INK2, style="italic")
top = (len(rows) - 1) * SP + 0.8
ax.text(6, top, "first year: models and website", ha="center", fontsize=7, color=INK2)
ax.text(18, top, "second year: website tables only", ha="center", fontsize=7, color=INK2)
ax.axvline(0, color=INK2, lw=0.8, ls="--", zorder=0)
ax.set_yticks([i * SP for i in range(len(rows))]); ax.set_yticklabels(rows)
ax.set_xlim(-1.5, 24.6); ax.set_ylim(-1.0, top + 0.35); ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])
ax.set_xlabel("Age (months after birth)")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
fig.tight_layout(); fig.savefig(OUT + "fig1_timeline.png", dpi=300); plt.close(fig)

# ================= Figure 2: the age-indexed front ends used by models and app
fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.4))
ax = axs[0, 0]
ax.plot(a, interp(ACU_BRAIN, a), color=C["vision"], label="brain response (VEP)")
ax.plot(a, interp(ACU_LOOK, a), color=C["vision"], ls=(0, (2, 1.5)), label="looking (acuity cards)")
ax.fill_between(a, interp(ACU_LOOK, a) / 2, interp(ACU_LOOK, a) * 2, color=C["vision"], alpha=0.12, lw=0)
ax.set_yscale("log"); ax.set_yticks([1, 3, 10, 30]); ax.set_yticklabels(["1", "3", "10", "30"])
ax.axhline(30, color=INK2, lw=0.8, ls=":"); ax.set_title("a  Acuity (cycles/degree)"); ax.legend(frameon=False, fontsize=6.5, loc="lower right")
ax = axs[0, 1]
ax.plot(a, interp(CS, a), color=C["vision"]); ax.set_yscale("log"); ax.set_yticks([2, 10, 100]); ax.set_yticklabels(["2", "10", "100"])
ax.set_title("b  Contrast sensitivity")
ax = axs[0, 2]
ax.plot(a, interp(SAT, a), color=C["vision"]); ax.set_ylim(0, 1.05); ax.set_title("c  Colour (× adult)")
ax = axs[1, 0]
ax.plot(a, interp(H_DB, a), color=C["hearing"]); ax.set_ylim(0, 27); ax.set_title("d  Extra quietness (dB)")
ax = axs[1, 1]
ax.plot(a, interp(H_TREBLE, a), color=C["hearing"]); ax.set_ylim(0, 13); ax.set_title("e  Treble cut, >3 kHz (dB)")
ax = axs[1, 2]
ax.plot(a, interp(H_STEREO, a), color=C["hearing"]); ax.set_ylim(0, 1.05); ax.set_title("f  Left-right cue (× adult)")
for ax in axs.flat:
    ax.set_xlim(0, 24); ax.set_xticks([0, 6, 12, 18, 24])
for ax in axs[1]: ax.set_xlabel("Age (months)")
fig.tight_layout(); fig.savefig(OUT + "fig2_frontends.png", dpi=300); plt.close(fig)

# ================= Figure 4: what the input allows vs what the network learned (5 seeds)
ex = json.load(open(REPO + "/results/extra.json"))
ceil, own = ex["ceiling"], ex["own_age"]["curriculum_only"]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
for ax, s, name in [(axs[0], "vision", "Vision"), (axs[1], "audio", "Hearing")]:
    ca = sorted(int(k) for k in ceil[s]); ax.plot(ca, [ceil[s][str(k)]["pos_r2"] for k in ca], color=INK2, ls=(0, (3, 2)), lw=1.5, label="input allows (supervised ceiling)")
    oa = sorted(int(k) for k in own[s]); m = np.array([np.mean([r["pos_r2"] for r in own[s][str(k)]]) for k in oa]); sd = np.array([np.std([r["pos_r2"] for r in own[s][str(k)]], ddof=1) for k in oa])
    col_ = C["vision"] if s == "vision" else C["hearing"]
    ax.fill_between(oa, m - sd, m + sd, color=col_, alpha=0.18, lw=0); ax.plot(oa, m, color=col_, marker="o", ms=4, label="world model learned (5 seeds)")
    ax.set_title(f"{'ab'[s=='audio']}  {name}: toy position (R²)"); ax.set_ylim(-0.05, 1.0); ax.set_xticks([0, 3, 6, 9, 12]); ax.set_xlabel("Age stage (months)")
axs[0].legend(frameon=False, fontsize=6.3, loc="lower right")
ax = axs[2]
ca = sorted(int(k) for k in ceil["touch"]); ax.plot(ca, [ceil["touch"][str(k)]["type_acc"] for k in ca], color=C["touch"], marker="o", ms=4)
ax.axhline(0.25, color=INK2, lw=0.8, ls=":"); ax.text(12, 0.27, "chance", ha="right", fontsize=6.5, color=INK2)
ax.set_ylim(0, 1.0); ax.set_xticks([0, 3, 6, 9, 12]); ax.set_title("c  Touch: identity (ceiling)"); ax.set_xlabel("Age stage (months)")
fig.tight_layout(); fig.savefig(OUT + "fig3_ceiling.png", dpi=300); plt.close(fig)

# ================= Figure 3: the app (cropped screenshots)
R = OUT + "raw/"  # full-page screenshots of the live site (docs/app, family view, light theme), 1280x900 at scale 2; not stored in the repo
if os.path.exists(R + "age3.png"):
    def crop(f, box): return Image.open(R + f).convert("RGB").crop(box)
    age = crop("age3.png", (248, 196, 2312, 1871))   # heading, age picker and sense cards at 3 months
    see = crop("see2.png", (248, 1030, 2312, 1847))   # adult view and 2-month view
    strip = crop("see2.png", (248, 2713, 2312, 3001)) # the same picture across the first year
    nums = crop("age3.png", (248, 2818, 2312, 3414))  # "The numbers behind this page", opened
    W = 2064
    def stack(ims, gap=40):
        H = sum(i.height for i in ims) + gap * (len(ims) - 1); c = Image.new("RGB", (W, H), "white"); y = 0
        for i in ims: c.paste(i, (0, y)); y += i.height + gap
        return c
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.6), gridspec_kw={"width_ratios": [1, 1]})
    axs[0].imshow(stack([age, nums])); axs[0].set_title("a  Age profile, 3 months")
    axs[1].imshow(stack([see, strip])); axs[1].set_title("b  See like a baby, 2 months")
    for ax in axs: ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT + "fig5_app.png", dpi=300); plt.close(fig)
print("ok")

# ================= Figure 4: experience-dependent signatures, redrawn from the companion studies' raw results
RES = os.path.join(REPO, "papers", "world-models", "companion") + "/"
arch = [r for r in json.load(open(RES + "cross-sense/results/arch.json")) if r["family"] == "gru"]
e2b = json.load(open(RES + "age-modeling/results/results_E2b.json"))
nar = [r for r in json.load(open(RES + "cross-sense/results/narrowing.json")) if r["family"] == "gru"]
SC = {"touch": C["touch"], "audio": C["hearing"], "vision": C["vision"]}
NAME = {"touch": "Touch", "audio": "Hearing", "vision": "Vision"}
fig = plt.figure(figsize=(7.2, 5.6))
gs = fig.add_gridspec(2, 6)
axa, axb, axc = fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6])
axd, axe = fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6])
for ax, met, title in [(axa, "id_acc", "a  Identity read-out"), (axb, "voe_teleport", "b  Surprise at a jump ($d_z$)")]:
    for s in ["touch", "audio", "vision"]:
        ms_ = [np.mean([r[met] for r in arch if r["sense"] == s and r["train_age"] == ag]) for ag in (0, 12)]
        ax.plot([0, 1], ms_, color=SC[s], marker="o", ms=5)
        ax.annotate(NAME[s], (1, ms_[1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7, color=INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["newborn\nfront end", "12-month\nfront end"]); ax.set_xlim(-0.2, 1.45); ax.set_title(title)
axa.set_ylim(0, 1.05); axb.set_ylim(0, 1.9)
starts = sorted({r["cfg"]["deficit_start"] for r in e2b})
vals = [[r["traj"][-1]["texture_only"] for r in e2b if r["cfg"]["deficit_start"] == st] for st in starts]
m = [np.mean(v) for v in vals]; sd = [np.std(v, ddof=1) for v in vals]
axc.errorbar(range(len(starts)), m, yerr=sd, color=C["vision"], marker="o", ms=5, capsize=0, lw=2)
axc.set_xticks(range(len(starts))); axc.set_xticklabels([f"{s}–{s+3}" for s in starts]); axc.set_ylim(0.6, 1.05)
axc.set_title("c  Blur window (months)"); axc.set_ylabel("Fine detail after recovery", fontsize=7.5)
def series(sense, sched, con, met):
    steps = sorted({r["step"] for r in nar if r["sense"] == sense})
    ys = np.array([[r[met] for r in nar if r["sense"] == sense and r["schedule"] == sched and r["contrast"] == con and r["step"] == st] for st in steps])
    return np.array(steps) + 1, ys.mean(1), ys.std(1, ddof=1)
for ax, sense, met, title in [(axd, "vision", "swap_voe", "d  Vision: surprise at an identity change"), (axe, "audio", "dprime", "e  Hearing: separability ($d'$)")]:
    for sched, con, lab, col, ls in [("mature", "native", "familiar pair", INK, "-"), ("mature", "nonnative", "unfamiliar pair", "#6250d6", "-"),
                                     ("exposed", "nonnative", "unfamiliar pair, exposed control", INK2, (0, (3, 2)))]:
        x, mu, s_ = series(sense, sched, con, met)
        ax.plot(x, mu, color=col, ls=ls, lw=1.8, label=lab); ax.fill_between(x, mu - s_, mu + s_, color=col, alpha=0.12, lw=0)
    ax.set_xscale("log"); ax.set_xlabel("Training step (one continuous run)"); ax.set_title(title)
axd.legend(frameon=False, fontsize=6.5, loc="upper left")
fig.tight_layout(); fig.savefig(OUT + "fig4_experience.png", dpi=300); plt.close(fig)
print("fig4 ok")
