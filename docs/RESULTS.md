# Results

The full write-up, with figures, methods and the companion studies, is the
paper-style report: [`report/REPORT.md`](../report/REPORT.md)
(LaTeX for Overleaf: [`report/report.tex`](../report/report.tex), with `report/refs.bib`).

Headline numbers (synthetic nursery, 3 seeds for the main regimes; see the paper for the current table):

| Finding | Number |
|---|---|
| Vision toy-position probe across checkpoints 0/3/6/12 mo, immature input | 0.02 / 0.36 / 0.77 / 0.86 |
| Same, adult-quality input from the start | 0.71 / 0.87 / 0.89 / 0.90 |
| Same, immature input + network growth | 0.05 / 0.04 / 0.58 / 0.80 |
| Hearing localisation at 12 mo, immature vs adult input | 0.45 ± 0.01 vs 0.37 ± 0.03 |
| Hidden-toy position at 12 mo: fused / vision memory / hearing | 0.77 / 0.64 / 0.48 |
| VoE index for a vanished toy, vision models (all regimes) | −0.17 to −0.06 (no expectation of reappearance) |
| VoE index, fused model (driven by hearing) | 0.21 → 0.30 |

Raw numbers: `results/summary.json` (means over seeds), `results/<regime>/seed*/metrics.json`,
`results/fusion_voe.json`. Regenerate figures with `python scripts/make_figures.py`.
