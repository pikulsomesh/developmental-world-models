# Results

The full write-up, with figures, methods and the companion studies, is the
paper-style report: [`papers/world-models/report/REPORT.md`](../papers/world-models/report/REPORT.md)
(arXiv-format LaTeX: [`papers/world-models/arxiv/main.tex`](../papers/world-models/arxiv/main.tex).)

Headline numbers (synthetic nursery, mean ± s.d. over 5 seeds per regime, as in the current paper):

| Finding | Number |
|---|---|
| Vision toy-position probe across checkpoints 0/3/6/12 mo, immature input | 0.02 / 0.31 / 0.74 / 0.85 |
| Same, adult-quality input from the start | 0.69 / 0.85 / 0.88 / 0.89 |
| Same, immature input + network growth | 0.04 / 0.04 / 0.53 / 0.79 |
| Hearing localisation at 12 mo, immature vs adult input | 0.45 ± 0.01 vs 0.39 ± 0.05 |
| Hidden-toy position at 12 mo: fused / vision memory / hearing | 0.75 ± 0.06 / 0.60 ± 0.08 / 0.42 ± 0.09 |
| VoE index for a vanished toy, vision models (all regimes) | −0.16 to −0.07 (no expectation of reappearance) |
| VoE index at 12 mo, fused model (carried by the hearing channel) | 0.32 ± 0.11 |

Raw numbers: `results/<regime>/seed*/metrics.json` (per seed; `results/summary.json` still holds the earlier 3-seed means),
`results/fusion_voe.json`. Regenerate figures with `python scripts/make_figures.py`.
