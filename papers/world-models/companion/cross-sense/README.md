# Cross-sense comparison experiments

Self-contained CPU experiments on how to compare developmental world models fairly across
vision, hearing and touch, and on extra model families (feed-forward, GRU, transformer,
predictive coding, JEPA). The write-up with figures is [`REPORT.md`](REPORT.md).

```bash
pip install torch numpy scipy scikit-learn pandas tabulate matplotlib
cd experiments/cross_sense
XSENSE_STEPS=1000 python experiments.py arch        # also: scale, curriculum, fused, narrowing, narrowing2, fused5
python analyze.py                                   # tables -> results/summary.md, figures -> results/figs/
```

| File | What it is |
|---|---|
| `xsense/world.py` | One latent world (moving object, 4 identities) rendered into vision, hearing and touch; violations applied to the latent |
| `xsense/age.py` | Per-sense, per-age sensory front ends (prenatal to 12 months) |
| `xsense/models.py` | Five world-model families with a matched core parameter budget |
| `xsense/run.py` | Training loop and the shared, sense-agnostic evaluation battery |
| `xsense/fused.py` | Fused multisensory model and sense-onset schedules |
| `xsense/narrowing.py` | Perceptual-narrowing test across checkpoints of one continuous run |
| `results/*.json` | Raw results behind every table and figure in the report |

A full run of all five suites takes about 2.5 hours on 4 CPU cores.
