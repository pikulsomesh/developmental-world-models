# Developmental World Models: one network per sense, grown across ages

Neural world models of sensory development in which **every sense has its own world model** (vision, hearing, touch, body position), each sense is fed through an **age-indexed filter** that makes its input as immature as a baby's at that age, and a **fusion model** combines them. The repository holds the code, trained checkpoints, two papers and the project site.

**Site and wiki (GitHub Pages):** <https://pikulsomesh.github.io/developmental-world-models/>. The home page is the *Baby Senses Lab*, an in-browser application for exploring what babies of each age typically sense; each paper has its own page.

## Two papers

| Paper | Focus | Folder |
|---|---|---|
| 1. *Learning Dynamics of Predictive World Models Under Developmental Input Constraints* | Computer science and theory: how staged, degraded input and network growth change what predictive networks learn, with four propositions and a controlled simulation | [`papers/world-models/`](papers/world-models/) |
| 2. Sensory perception across infant ages and the Baby Senses Lab | Developmental neuroscience: what infants of each age perceive through each sense, and how age-tuned models simulate it | [`papers/baby-senses/`](papers/baby-senses/) |

Paper 1 sources: [`report/`](papers/world-models/report/) (Markdown template, figures, built PDF, internal referee report, citation audit) and [`arxiv/`](papers/world-models/arxiv/) (arXiv-format LaTeX, bibliography). Companion experiments are in [`companion/`](papers/world-models/companion/).

![What each sense delivers at each age](results/figures/fig_senses_by_age.png)

## What is here

| Path | What it is |
|---|---|
| [`docs/PLAN.md`](docs/PLAN.md) | The research plan: what changes from the source proposal and why, per-sense specs, growth and fusion design, hypotheses, roadmap, paper plan |
| [`papers/world-models/report/REPORT.md`](papers/world-models/report/REPORT.md) | **Paper 1** with figures (PDF: `report.pdf`; arXiv-format LaTeX: `papers/world-models/arxiv/main.tex`) |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Headline numbers |
| `dwm/capability.py` | Per-sense developmental tables (acuity, contrast, colour, auditory resolution, tactile acuity, proprioceptive noise) and the capacity schedule |
| `dwm/env.py` | A synthetic multisensory "nursery": a toy that is seen, heard and felt, the infant's babbling hand, occluders, violation-of-expectation episodes |
| `dwm/senses.py` | Age-dependent sensory front ends (non-learned filters) |
| `dwm/models.py` | `SenseWorldModel` (encoder → windowed dynamics → decoder) with **function-preserving growth**; `FusionWorldModel` |
| `dwm/train.py`, `dwm/fusion.py`, `scripts/analyze_fusion_voe.py` | Training, linear probes, VoE surprise, fusion experiments |
| `scripts/run_experiments.py` | Runs every sense × condition × age stage, then fusion |
| `scripts/make_figures.py` | Merges results and draws the figures |
| `models/` | Trained checkpoints (curriculum_only, adult_scratch, grown; seed 0) |
| `papers/` | The two papers: `world-models/` (theory and experiments) and `baby-senses/` (sensory perception) |
| `docs/` | Project site and wiki (GitHub Pages) |
| `tests/` | Unit test that growth leaves the model's function unchanged |

## Quick start (CPU only)

```bash
pip install -r requirements.txt
python -m pytest -q                                   # growth is function-preserving
python scripts/run_experiments.py --quick --out /tmp/q # ~2 min smoke test
# full grid of regimes and seeds:
scripts/queue.sh
python scripts/analyze_fusion_voe.py --seeds 0,1,2
python scripts/analyze_spectral.py
python scripts/make_paper_figures.py && papers/world-models/report/build.sh
```

## The idea in one picture

```
age ─► sensory front end F_s(age) ─► encoder ─► z_t ─► dynamics over last k(age) latents + action ─► ẑ_{t+1}
            (blur, contrast,            ▲                      ▲
             colour, freq. res., …)     └── width w(age) ──────┘   grows 0→3→6→12 months, function-preserving

per-sense latents z^vision, z^audio, z^touch, z^proprio ─► fusion world model (modality dropout, cross-modal fill-in)
```

## Status

Synthetic scene, four senses, six training regimes, fusion world model, own-age and adult-input probes, information ceilings, violation-of-expectation events and statistics are all included. Planned next: staged-onset fusion, realistic optics, MIMo embodiment, SAYCam/BabyView. See the roadmap in the plan.

## License

MIT (code). The plan and paper text are © the authors until the preprint is posted.
