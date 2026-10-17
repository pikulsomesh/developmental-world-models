"""Fit published infant-minus-adult pure-tone threshold differences against age.
Run: python3 fit.py  (reads data.csv; writes fit.json, fig_hearing_fit.png)
Missing: no data at birth, none between 12 and ~60 months for absolute pure-tone thresholds
(see README). Values outside 1-12 months are extrapolation and are flagged as such."""
import csv, json, numpy as np
from scipy.optimize import curve_fit
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("data.csv")))
CUR_Q = {0: 25, 3: 20, 6: 12, 12: 6, 24: 3}
CUR_T = {0: 12, 1: 6, 3: 4, 6: 1, 12: 0}
def cur(tbl, a):
    k = sorted(tbl); return float(np.interp(a, k, [tbl[x] for x in k]))
AGES = [0, 1, 3, 6, 12, 24]

def overall(flag):
    """mean of low and high difference per age; 12 months = pooled value."""
    out = {}
    for age in (1, 3, 6):
        lo = [float(r["infant_minus_adult_db"]) for r in rows if r[flag] == "1" and int(r["age_months"]) == age and r["kind"] == "low"]
        hi = [float(r["infant_minus_adult_db"]) for r in rows if r[flag] == "1" and int(r["age_months"]) == age and r["kind"] == "high"]
        out[age] = (np.mean(lo) if lo else np.nan, np.mean(hi))
    return out
def build(flag):
    lh = overall(flag)
    ages = [1, 3, 6, 12]
    ov = [np.nanmean(lh[a]) for a in (1, 3, 6)] + [12.5]   # 1 month: 4 kHz only (500 Hz value not verifiable)
    hl = {a: lh[a][1] - lh[a][0] for a in lh if not np.isnan(lh[a][0])}
    return ages, ov, hl, lh

def expf(t, a, tau, c): return a*np.exp(-t/tau) + c
out = {}
for name, flag in (("primary", "use_primary"), ("alt", "use_alt")):
    ages, ov, hl, lh = build(flag)
    p, _ = curve_fit(expf, ages, ov, p0=[40, 3, 12], bounds=([0, 0.5, 0], [200, 50, 30]))
    pl = {a: float(np.interp(a, ages, ov)) for a in AGES}   # clamps outside 1-12 mo
    ex = {a: float(expf(a, *p)) for a in AGES}
    out[name] = dict(
        low_high_by_age={a: [float(x) for x in lh[a]] for a in lh},
        overall_points=dict(zip(map(str, ages), map(float, ov))),
        high_minus_low_by_age={str(a): float(v) for a, v in hl.items()},
        exp_params=dict(a=float(p[0]), tau_months=float(p[1]), c=float(p[2])),
        exp_fit={str(a): ex[a] for a in AGES},
        exp_fit_extrapolated_ages=[0, 24],
        pwl_interp={str(a): pl[a] for a in AGES},
        pwl_note="piecewise-linear; ages 0 and 24 are clamped to nearest datum (not data)",
        current_quieter={str(a): cur(CUR_Q, a) for a in AGES},
        diff_fit_minus_current={str(a): ex[a] - cur(CUR_Q, a) for a in AGES},
        current_treble={str(a): cur(CUR_T, a) for a in AGES},
    )
# treble: high-minus-low from behavioural data; 8 kHz variant at 3 months
hl8 = float([r for r in rows if r["id"] == "b4"][0]["infant_minus_adult_db"]) - 23.0  # 8 kHz minus 500 Hz, Olsho
out["treble_8k_3mo_minus_500hz"] = hl8
json.dump(out, open("fit.json", "w"), indent=2)

# figure
plt.rcParams.update({"font.size": 7, "axes.facecolor": "white"})
fig, ax = plt.subplots(figsize=(3.5, 2.6), facecolor="white")
ax.grid(color="#e4e3df", lw=0.6); ax.set_axisbelow(True)
t = np.linspace(0, 24, 200)
P = out["primary"]["exp_params"]
ax.plot(t[(t >= 1) & (t <= 12)], expf(t[(t >= 1) & (t <= 12)], P["a"], P["tau_months"], P["c"]), color="#eb6834", lw=1.6, label="Fit (mean low/high), data span")
ax.plot(t[t <= 1], expf(t[t <= 1], P["a"], P["tau_months"], P["c"]), color="#eb6834", lw=1.0, ls=":")
ax.plot(t[t >= 12], expf(t[t >= 12], P["a"], P["tau_months"], P["c"]), color="#eb6834", lw=1.0, ls=":")
ks = sorted(CUR_Q); ax.plot(ks, [CUR_Q[k] for k in ks], color="#52514e", ls="--", lw=1.4, label="Current anchors")
mk = {"Olsho": ("o", "Olsho et al. 1988"), "Werner 2007": ("s", "Werner 2007"), "Werner & G": ("^", "Werner & Gillenwater 1990 (secondary)")}
seen = set()
for r in rows:
    if r["use_primary"] != "1" and r["use_alt"] != "1": continue
    if r["kind"] == "overall": x, y, key = 12, 12.5, "Olsho"
    else: x, y = int(r["age_months"]), float(r["infant_minus_adult_db"]); key = "Olsho" if r["source"].startswith("Olsho") else ("Werner & G" if r["source"].startswith("Werner & G") else "Werner 2007")
    m, lab = mk[key]; fc = "#2a2a28" if r["kind"] == "low" else "white"
    if r["kind"] == "overall": fc = "#9a9a96"
    ax.scatter([x], [y], marker=m, s=18, facecolor=fc, edgecolor="#2a2a28", lw=0.7, zorder=3, label=None)
    seen.add(key)
for k in seen: ax.scatter([], [], marker=mk[k][0], s=18, facecolor="white", edgecolor="#2a2a28", lw=0.7, label=mk[k][1])
ax.scatter([], [], marker="o", s=18, facecolor="#2a2a28", edgecolor="#2a2a28", label="filled: low freq; open: high")
ax.set_xlabel("Age (months)"); ax.set_ylabel("Infant minus adult threshold (dB)")
ax.set_xlim(0, 24); ax.set_ylim(0, 50)
ax.legend(fontsize=5.5, frameon=False, loc="upper right")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig("fig_hearing_fit.png", dpi=300, facecolor="white")
print(json.dumps(out, indent=1))
