# Follow-up runs for paper 2: training to 24 months, and latent versus pixel prediction

`run.py` reuses the main-study code in `dwm/` and `scripts/analyze_extra.py` without changing it.

```
python run.py --job vision_vep --seed 0       # jobs: vision_vep, vision_look, audio, vision_jepa, vision_jepa_vr
python run.py --ceiling vision_look           # fronts: vision_vep, vision_look, audio
./queue.sh                                    # every job, 3 seeds, three processes at a time
python make_fig7.py                           # figures/fig7_beyond12_jepa.png and results/summary.json
```

CPU only, one thread per job; each vision job takes 1-2 hours. Results (5 seeds, own-age probe, toy-position R^2):

| | 12 mo | 18 mo | 24 mo | ceiling at 24 mo |
|---|---|---|---|---|
| vision, evoked-potential acuity | 0.85 ± 0.04 | 0.85 ± 0.05 | 0.88 ± 0.02 | 0.95 |
| vision, behavioural acuity | 0.82 ± 0.06 | 0.83 ± 0.08 | 0.86 ± 0.05 | 0.94 |
| hearing | 0.45 ± 0.01 | 0.45 ± 0.01 | 0.45 ± 0.01 | 0.49 |

Vision, 0-12 months, position R^2 at 12 months: pixel prediction 0.85 ± 0.04; latent prediction with an EMA target -0.00 (collapsed, per-dimension s.d. ≤ 0.006); with VICReg terms 0.02 ± 0.04 (not collapsed, effective dimension 14.7 against 6.8 for pixel prediction).

Seeds 0-2 were first run as three-seed jobs; seeds 3-4 were added later, and the vision_jepa_vr runs were re-run by `scripts/run_future_work.py --job voe_jepa_vr` (same training, plus surprise measurements), which overwrote their result files. The plain latent objective collapsed in a smoke test, which is why the VICReg variant was added. `results/*.json` hold every stage; `logs/` are not kept.
