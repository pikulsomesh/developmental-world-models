"""Summarise results/future_work/*.json (written by scripts/run_future_work.py) into
results/future_work/summary.json, the numbers quoted in paper 1."""
import glob, json, os
import numpy as np

R = os.path.join(os.path.dirname(__file__), "..", "results")
FW = os.path.join(R, "future_work")


def ms(x):
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0, "n": len(x), "values": x.round(4).tolist()}


def load(job):
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(FW, f"{job}_seed[0-9].json")))]


out = {}
# experience-matched vs staged (staged = curriculum_only, own-age probe, from results/extra.json)
extra = json.load(open(os.path.join(R, "extra.json")))
for sense, key in [("vision", "matched_vision"), ("audio", "matched_audio")]:
    runs = load(key)
    ages = [c["age"] for c in runs[0]["checkpoints"]]
    out[key] = {str(a): ms([r["checkpoints"][i]["probe_own"]["pos_r2"] for r in runs]) for i, a in enumerate(ages)}
    out[key + "_type_acc"] = {str(a): ms([r["checkpoints"][i]["probe_own"]["type_acc"] for r in runs]) for i, a in enumerate(ages)}
    st = extra["own_age"]["curriculum_only"][sense]  # {age: [per seed probe dict]}
    out["staged_" + sense] = {a: ms([p["pos_r2"] for p in v]) for a, v in st.items()}
# violation-of-expectation
for job, kinds in [("voe_gauss", ["voe_nll", "voe_mse"]), ("voe_jepa_vr", ["voe_latent"])]:
    runs = load(job)
    ages = [s["age"] for s in runs[0]["stages"]]
    out[job] = {"probe_pos_r2": {str(a): ms([r["stages"][i]["probe_own"]["pos_r2"] for r in runs]) for i, a in enumerate(ages)}}
    for k in kinds:
        out[job][k] = {str(a): ms([r["stages"][i][k]["voe_index"] for r in runs]) for i, a in enumerate(ages)}
        out[job][k + "_n_positive_at_12"] = int(sum(r["stages"][-1][k]["voe_index"] > 0 for r in runs))
# main-study reference (pixel forecaster, unimodal vision, curriculum_only)
ref = {}
for s in range(5):
    p = os.path.join(R, "curriculum_only", f"seed{s}", "metrics.json")
    if os.path.exists(p):
        m = json.load(open(p))["senses"]["vision"]["curriculum_only"]
        for i, e in enumerate(m):
            ref.setdefault(str(e["age"]), []).append(e["voe"]["voe_index"])
out["voe_pixel_reference"] = {a: ms(v) for a, v in ref.items()}
# spectral
sp = json.load(open(os.path.join(FW, "spectral.json")))
out["spectral"] = {"input": sp["input"]}
for regime in ("curriculum_only", "adult_scratch"):
    out["spectral"][regime] = {}
    for age in ("0", "3", "6", "12"):
        cells = [sp[regime][s][age] for s in sp[regime] if age in sp[regime][s]]
        out["spectral"][regime][age] = {k: ms([c[k] for c in cells]) for k in ("eff_rank", "top10", "top100", "jac_norm")}
json.dump(out, open(os.path.join(FW, "summary.json"), "w"), indent=1)


def show(d, depth=0):
    for k, v in d.items():
        if isinstance(v, dict) and "mean" in v:
            print("  " * depth + f"{k}: {v['mean']:.3f} ± {v['sd']:.3f} (n={v['n']})")
        elif isinstance(v, dict):
            print("  " * depth + str(k)); show(v, depth + 1)
        else:
            print("  " * depth + f"{k}: {v}")


show(out)
