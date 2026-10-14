# Developmental World Models — one per sense, grown across ages

Neural "world models" of infant perception in which **every sense has its own world
model** (vision, audition, touch, proprioception), each sense **matures on its own
timetable**, each model **grows in place** from a newborn-sized network to a 12-month
one, and a **fusion world model** combines them.

This repository is the Phase-0 prototype of the research plan in
[`docs/PLAN.md`](docs/PLAN.md). The plan builds on the proposal *Developmental World Models:
Neural Network Architectures for Simulating Infant Cognition* (Dec 2025). The aim is an arXiv
preprint; the current draft is the Phase-0 report in [`report/`](report/) (`paper/refs.bib` holds the shared bibliography).

![What each sense delivers at each age](results/figures/fig_senses_by_age.png)

## What is here

| Path | What it is |
|---|---|
| [`docs/PLAN.md`](docs/PLAN.md) | The research plan: what changes from the source proposal and why, per-sense specs, growth and fusion design, hypotheses, roadmap, paper plan |
| [`report/REPORT.md`](report/REPORT.md) | **Paper-style report** of Phase 0 with figures (LaTeX version `report/report.tex` for Overleaf) |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Headline numbers |
| `dwm/capability.py` | Per-sense developmental tables (acuity, contrast, colour, auditory resolution, tactile acuity, proprioceptive noise) and the capacity schedule |
| `dwm/env.py` | A synthetic multisensory "nursery": a toy that is seen, heard and felt, the infant's babbling hand, occluders, violation-of-expectation episodes |
| `dwm/senses.py` | Age-dependent sensory front ends (non-learned filters) |
| `dwm/models.py` | `SenseWorldModel` (encoder → windowed dynamics → decoder) with **function-preserving growth**; `FusionWorldModel` |
| `dwm/train.py`, `dwm/fusion.py`, `scripts/analyze_fusion_voe.py` | Training, linear probes, VoE surprise, fusion experiments |
| `scripts/run_experiments.py` | Runs every sense × condition × age stage, then fusion |
| `scripts/make_figures.py` | Merges results and draws the figures |
| `tests/` | Unit test that growth leaves the model's function unchanged |

## Quick start (CPU only)

```bash
pip install -r requirements.txt
python -m pytest -q                                   # growth is function-preserving
python scripts/run_experiments.py --quick --out /tmp/q # ~2 min smoke test
# full Phase-0 grid (14 runs, ~5 h on 4 cores):
scripts/queue.sh
python scripts/analyze_fusion_voe.py --seeds 0,1,2
python scripts/make_figures.py && report/build.sh
```

## The idea in one picture

```
age ─► sensory front end F_s(age) ─► encoder ─► z_t ─► dynamics over last k(age) latents + action ─► ẑ_{t+1}
            (blur, contrast,            ▲                      ▲
             colour, freq. res., …)     └── width w(age) ──────┘   grows 0→3→6→12 months, function-preserving

per-sense latents z^vision, z^audio, z^touch, z^proprio ─► fusion world model (modality dropout, cross-modal fill-in)
```

## Status

Phase 0 (this repo): synthetic data, four senses, six training regimes, fusion — done,
three seeds for the main regimes. Next: more seeds, staged-onset fusion, real optics, MIMo embodiment,
SAYCam/BabyView. See the roadmap in the plan.

## License

MIT (code). The plan and paper text are © the authors until the preprint is posted.
