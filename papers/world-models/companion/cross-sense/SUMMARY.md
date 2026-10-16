# Cross-sense comparison: short summary for the end-of-day report

Full write-up: `REPORT.md` (same folder). Figures: `figures/*.png`. Raw results: `results/*.json`. Code: `xsense/`, also on branch `cross-sense-comparison` under `experiments/cross_sense/`.

**Method.** One latent toy world (an object of 4 identities moving and bouncing) is rendered into vision, hearing and touch, and every violation-of-expectation event is applied to the latent before rendering, so the task is identical across senses. Five world-model families (feed-forward, GRU, transformer, predictive coding, JEPA) at a matched ~20k core parameters, per-sense age front ends from prenatal to 12 months, about 290 CPU runs with 2–5 seeds.

**Findings.**
1. Per-sense front ends alone reproduce the infant order at birth: touch near mature, hearing partly, vision far behind (vision permanence R² −2.23 at birth vs 0.33 at 12 mo; touch 0.38 vs 0.32). `figures/heterochrony.png`
2. Memory is the only sense-general requirement: the memoryless model is last on teleport VoE in all three senses. The ranking of the other families changes by sense, so use one family for all senses. `figures/arch_12mo.png`
3. The objective decides what a model notices: JEPA tracks hidden objects best (vision 0.70 vs GRU 0.33) but is nearly blind to identity changes in sound (VoE d_z 0.08 vs 1.29). Predictive coding beats the GRU at equal size in vision (0.52 vs 0.33).
4. Minimal size differs by sense (hearing ~6k core parameters, touch ~24k, vision still improving at 97k), and extra capacity does not rescue newborn vision (permanence −2.67 at 1.7k, −2.09 at 97k). Front ends, not smaller models, carry age. `figures/capacity.png`
5. Blurry-first curricula help hidden-object tracking in vision (0.51 vs 0.33) at a cost in sharp prediction, and change nothing in hearing or touch. `figures/curricula.png`
6. In a fused model the sense that comes online last is predicted worst on its own (vision alone R² −0.18 when last vs 0.43 when first, 5 seeds), but onset order does not change how well the shared state tracks hidden objects (0.60–0.62). A hint that the human order helps fill in vision from hearing and touch did not survive 5 seeds (0.10 ± 0.15). `figures/onset_order.png`
7. Perceptual narrowing emerges from plain next-step prediction on a restricted diet: in vision, surprise at a never-seen contrast rises to 0.54 then falls to 0.29 while the native contrast reaches 1.18 and an exposed control reaches 1.17. Hearing narrows in representation (d′ 5.19 → 1.33) but never shows surprise. Both signatures replicate with predictive-coding and transformer models. `figures/narrowing.png`

**Recommendations.** Latent-matched paradigms for every cross-sense claim; match core parameters and objective, report adapters and experience hours per sense; always report one reconstruction and one latent-prediction model; add predictive coding to the model zoo; make perceptual narrowing the flagship cross-sense test and pre-register which measure counts as discrimination; headline metric = Kendall τ between model and infant milestone order.

**Caveat.** Toy setting; front-end tables are coarse literature anchors. Evidence about fair comparison design, not about infants.
