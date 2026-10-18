# Follow-up runs for paper 2: training to 24 months, and latent versus pixel prediction

`run.py` reuses the main-study code in `dwm/` and `scripts/analyze_extra.py` without changing it.

```
python run.py --job vision_vep --seed 0       # jobs: vision_vep, vision_look, audio, vision_jepa, vision_jepa_vr
python run.py --ceiling vision_look           # fronts: vision_vep, vision_look, audio
./queue.sh                                    # every job, 3 seeds, three processes at a time
python make_fig7.py                           # figures/fig7_beyond12_jepa.png and results/summary.json
```

CPU only, one thread per job; each vision job takes 1-2 hours. Results (3 seeds, own-age probe, toy-position R^2):

| | 12 mo | 18 mo | 24 mo | ceiling at 24 mo |
|---|---|---|---|---|
| vision, evoked-potential acuity | 0.86 ± 0.03 | 0.88 ± 0.02 | 0.89 ± 0.02 | 0.95 |
| vision, behavioural acuity | 0.80 ± 0.05 | 0.79 ± 0.07 | 0.83 ± 0.04 | 0.94 |
| hearing | 0.45 ± 0.01 | 0.45 ± 0.01 | 0.44 ± 0.01 | 0.49 |

Vision, 0-12 months, position R^2 at 12 months: pixel prediction 0.86 ± 0.03; latent prediction with an EMA target 0.00 (collapsed, per-dimension s.d. ≤ 0.006); with VICReg terms 0.03 ± 0.04 (not collapsed, effective dimension 15 against 7 for pixel prediction).

The plain latent objective collapsed in a smoke test, which is why the VICReg variant was added. `results/*.json` hold every stage; `logs/` are not kept.
