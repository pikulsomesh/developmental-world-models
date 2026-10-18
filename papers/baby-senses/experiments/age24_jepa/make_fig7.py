"""Fig. 7 of paper 2 and a summary table, from results/*.json written by run.py."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "..", "..", "figures", "fig7_beyond12_jepa.png")
C = {"vision": "#2a78d6", "look": "#7fb2ec", "hearing": "#eb6834", "jepa": "#6250d6", "jepa_vr": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 2})


def load(job):
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(R, f"{job}_seed[0-9].json")))]
    ages = [s["age"] for s in runs[0]["stages"]]
    runs = [r for r in runs if len(r["stages"]) == len(ages)]

    def get(key, sub):
        return np.array([[s[key][sub] for s in r["stages"]] for r in runs])
    return ages, get("probe_own", "pos_r2"), get("spread", "eff_dim"), get("spread", "z_std"), get("probe_own", "type_acc")


def ceil(front):
    c = json.load(open(os.path.join(R, f"ceiling_{front}.json")))
    a = sorted(int(k) for k in c)
    return a, [c[str(k)]["pos_r2"] for k in a]


summary = {}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
ax1.axvspan(12, 24, color="#f2f1ed", zorder=0, lw=0)
for job, front, col, lab in [("vision_vep", "vision_vep", C["vision"], "vision, evoked-potential acuity"),
                             ("vision_look", "vision_look", C["look"], "vision, behavioural acuity"),
                             ("audio", "audio", C["hearing"], "hearing")]:
    ages, r2, ed, zs, acc = load(job)
    m, s = r2.mean(0), r2.std(0)
    ax1.errorbar(ages, m, yerr=s, color=col, marker="o", ms=4, capsize=2, label=lab)
    ca, cv = ceil(front)
    ax1.plot(ca, cv, color=col, ls=(0, (3, 2)), lw=1.2)
    summary[job] = {"ages": ages, "n_seeds": len(r2), "pos_r2_mean": m.round(3).tolist(), "pos_r2_sd": s.round(3).tolist(),
                    "type_acc_mean": acc.mean(0).round(3).tolist(), "eff_dim_mean": ed.mean(0).round(2).tolist(),
                    "ceiling_pos_r2": dict(zip(map(str, ca), np.round(cv, 3).tolist()))}
ax1.set_xticks([0, 3, 6, 12, 18, 24]); ax1.set_xlabel("age stage (months)"); ax1.set_ylabel("toy position, $R^2$")
ax1.set_ylim(-0.05, 1.0); ax1.set_title("a  Training into the second year")
ax1.plot([], [], color=INK2, ls=(0, (3, 2)), lw=1.2, label="input allows (supervised ceiling)")
ax1.legend(frameon=False, fontsize=6.8, loc="lower right")

for job, col, lab in [("vision_vep", C["vision"], "pixel prediction"), ("vision_jepa", C["jepa"], "latent prediction (EMA target)"),
                      ("vision_jepa_vr", C["jepa_vr"], "latent prediction + VICReg")]:
    ages, r2, ed, zs, acc = load(job)
    k = [i for i, a in enumerate(ages) if a <= 12]
    ages, r2, ed, zs = [ages[i] for i in k], r2[:, k], ed[:, k], zs[:, k]
    ax2.errorbar(ages, r2.mean(0), yerr=r2.std(0), color=col, marker="o", ms=4, capsize=2, label=lab)
    summary.setdefault(job + "_0to12", {}).update({"ages": ages, "n_seeds": len(r2), "pos_r2_mean": r2.mean(0).round(3).tolist(),
                                                   "pos_r2_sd": r2.std(0).round(3).tolist(), "eff_dim_mean": ed.mean(0).round(2).tolist(),
                                                   "z_std_mean": zs.mean(0).round(4).tolist()})
ca, cv = ceil("vision_vep")
ax2.plot([a for a in ca if a <= 12], [v for a, v in zip(ca, cv) if a <= 12], color=INK2, ls=(0, (3, 2)), lw=1.2, label="input allows")
ax2.set_xticks([0, 3, 6, 12]); ax2.set_xlabel("age stage (months)"); ax2.set_ylabel("toy position, $R^2$")
ax2.set_ylim(-0.05, 1.0); ax2.set_title("b  Predicting pixels or latents")
ax2.legend(frameon=False, fontsize=6.8, loc="center right", bbox_to_anchor=(1.0, 0.3))
fig.tight_layout(); fig.savefig(FIG, dpi=300); plt.close(fig)
json.dump(summary, open(os.path.join(R, "summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
