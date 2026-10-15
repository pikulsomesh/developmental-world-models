"""Fill report/REPORT.template.md with numbers from results/ and write report/REPORT.md.

Placeholders are {{expr}}, evaluated with the helpers below, so every number in the paper is
regenerated from the results files when seeds are added:

  m(regime, sense, key, sub, stage=3, sd=True)  mean (± s.d.) over seeds of a checkpoint metric
  f(group, name, metric, age=12, sd=True)       fusion metric (grown regime)
  fv(event, channel, age=12, field="voe_index") fused VoE over seeds
  ce(sense, metric, age)                         information ceiling
  oa(regime, sense, age, metric="pos_r2")        own-age probe
  st(name, field)                                statistics of a planned comparison
  n(regime, sense="vision")                      number of seeds
  cp(sense, field)                               compute accounting
  ms(field, sense=None)                          milestones
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
STAGES = [0, 3, 6, 12]

RUNS = {}
for p in glob.glob(os.path.join(RES, "*", "seed*", "metrics.json")):
    RUNS[(p.split(os.sep)[-3], int(p.split(os.sep)[-2][4:]))] = json.load(open(p))
EXTRA = json.load(open(os.path.join(RES, "extra.json"))) if os.path.exists(os.path.join(RES, "extra.json")) else {}
FV = json.load(open(os.path.join(RES, "fusion_voe.json"))) if os.path.exists(os.path.join(RES, "fusion_voe.json")) else {}


def fmt(v, sd=True, d=2):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return "n/a"
    s = f"{v.mean():.{d}f}"
    if sd and len(v) > 1:
        s += f" ± {v.std(ddof=1):.{d}f}"
    return s.replace("-", "−")


def _vals(regime, sense, key, sub, stage):
    out = []
    for (rg, _), r in sorted(RUNS.items()):
        t = r["senses"].get(sense, {}).get(regime) if rg == regime else None
        if t:
            out.append(t[stage][key][sub] if sub else t[stage][key])
    return out


def m(regime, sense, key, sub=None, stage=3, sd=True, d=2):
    return fmt(_vals(regime, sense, key, sub, stage), sd, d)


def n(regime, sense="vision"):
    return str(len(_vals(regime, sense, "probe_adult", "pos_r2", 3)))


def f(group, name, metric, age=12, sd=True):
    v = [r["fusion"][str(age)][group][name][metric] for (rg, _), r in sorted(RUNS.items()) if rg == "grown" and r.get("fusion")]
    return fmt(v, sd)


def fv(event, channel, age=12, field="voe_index", sd=True):
    v = [FV[s][str(age)][event][channel][field] for s in FV if event in FV[s].get(str(age), {})]
    return fmt(v, sd)


def ce(sense, metric, age):
    return fmt([EXTRA["ceiling"][sense][str(age)][metric]], False)


def oa(regime, sense, age, metric="pos_r2", sd=True):
    return fmt([d[metric] for d in EXTRA.get("own_age", {}).get(regime, {}).get(sense, {}).get(str(age), [])], sd)


def st(name, field):
    c = EXTRA["stats"]["comparisons"][name]
    if field == "p":
        return f"{c['p']:.2f}" if c["p"] >= 0.01 else f"{c['p']:.3f}"
    if field == "g":
        return f"{c['hedges_g']:.1f}".replace("-", "−")
    if field == "n":
        return f"{len(c['a'])}/{len(c['b'])}"
    return str(c[field])


def cp(sense, field):
    v = EXTRA["compute"][sense][field]
    return f"{v:.2f}" if isinstance(v, float) and v < 10 else f"{v:,}"


def ms(field, sense=None):
    x = EXTRA["milestones"][field]
    return f"{x[sense]}" if sense else (f"{x:.2f}" if isinstance(x, float) else str(x))


def main():
    src = open(os.path.join(ROOT, "report", "REPORT.template.md")).read()
    ctx = {k: v for k, v in globals().items() if callable(v)}

    def sub(mo):
        try:
            return str(eval(mo.group(1), ctx))
        except Exception as e:  # keep going so missing pieces are visible
            print("could not fill", mo.group(1), "->", repr(e))
            return "[missing]"

    out = re.sub(r"\{\{(.+?)\}\}", sub, src)
    open(os.path.join(ROOT, "report", "REPORT.md"), "w").write(out)
    print("wrote report/REPORT.md")


if __name__ == "__main__":
    main()
