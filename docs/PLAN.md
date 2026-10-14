# Research plan: Sensory-specific Developmental World Models

*Status: living document. Phase 0 (a minimal, CPU-scale prototype) is implemented in this
repository; see the [Phase-0 report](../report/REPORT.md) for what it actually shows.*

## 0. In one paragraph

The source proposal ("Developmental World Models: Neural Network Architectures for
Simulating Infant Cognition", Dec 2025) builds **one vision-centred world model per age**
(0–2, 3–6, 7–12, 12–24 months) and validates it against behaviour (BIB, violation of
expectation) and infant fMRI. This plan keeps its goals and its capability matrix but
changes the unit of modelling: **one world model per sense** (vision, audition, touch,
proprioception), each with its *own* maturation timetable, each **grown** in place from a
newborn-sized network to an older one (rather than re-built per age), and then
**combined** by a fusion world model whose cross-modal binding is itself developmental.
The scientific bet is the old "starting small" / heterochrony idea (Turkewitz & Kenny,
1982; Elman, 1993): senses that come online at different times, with low-fidelity input
and small capacity at first, should produce *better organised and more robust*
representations than an adult-from-scratch model, and multisensory redundancy should
scaffold capacities such as object permanence before any single sense supports them.

---

## 1. What we keep from the source proposal, and what we change

| Topic | Source proposal | This plan | Why |
|---|---|---|---|
| Unit of modelling | One model per age, vision first; audio/motor as later add-ons | One world model **per sense**, each with its own age tables; a separate fusion model | Senses mature on different timetables (touch and hearing are functional before birth, vision is the least mature at birth). A single age knob cannot express that. |
| Moving between ages | Separate architectures per age (`_build_minimal_encoder`, … `resnet50`) | **Grow** one network: function-preserving widening + longer memory window at each stage | A real infant is one system that keeps its past learning. Growing lets us ask whether *history* matters (grown vs adult-from-scratch). |
| 6–24 month encoders | ImageNet-pretrained ResNet-18/50 | No adult-pretrained weights anywhere | ImageNet features encode adult-scale, labelled experience; they break the developmental constraint the project is about. |
| Contrast model | Multiply all contrast by CS/100 (newborn keeps 2% of contrast) | Log-compressed contrast gain (newborn ≈ 15%), CSF-shaped filter as a later upgrade | Contrast *sensitivity* is a detection threshold, not a global gain; the proposal's version renders almost every newborn image blank. |
| Colour at birth | Saturation 0.0 (fully achromatic) | 0.1, rising to 1.0 by 5 months | Newborns show some red/green discrimination (Adams et al.); "fully achromatic" overstates it. |
| Acuity → blur | `sigma = 0.5 * 30 / cpd` pixels, independent of viewing distance (the function's distance/size arguments are unused) | Same law, rescaled to image size; Phase 1 replaces it with a proper degrees-of-visual-angle conversion using distance and stimulus size | Makes the parameters it already takes meaningful. |
| Acuity values | Table mixes VEP (≈15–20 cpd at 6 mo) and behavioural (Teller cards, ≈6–9 cpd at 6 mo) values | Keep both columns explicitly; default to behavioural for "what the infant can use", VEP as an upper bound | They differ 2–3×, which changes every downstream "can see it?" answer. |
| Memory | LSTM with a Python list buffer inside `forward` | Explicit **window** of past latents that grows with age (1 → 2 → 4 → 6 steps in the prototype) | Stateless, batchable, and its growth is exactly function-preserving. |
| Training objective | Next-latent MSE only | Reconstruction + next-observation prediction + next-latent consistency | Pure latent prediction collapses to a constant; the decoder keeps latents honest. |
| Validation | BIB, VoE, infant fMRI | Same, plus per-sense benchmarks and *cross-modal* tests (fill-in, occluded tracking, McGurk-style conflict) | The sensory-specific framing needs tests that only a combined model can pass. |
| Data | SAYCam for everything | Synthetic multisensory nursery → MIMo (touch, proprioception, vestibular) → SAYCam / BabyView (vision + audio) → HomeBank / CHILDES (speech) | No natural dataset contains infant touch or proprioception; MIMo does. |

Everything else in the proposal (the capability matrix in §3, the activity library, the
Streamlit app, the fMRI encoding-model method) is kept and becomes Phase 4–5 work.

---

## 2. Architecture

```
            age a (months)
                │  sets: sensory filter params, width w(a), memory window k(a)
                ▼
  ┌──────────── per-sense world models (trained separately, grown in place) ───────────┐
  │ vision  o^v ─► F_v(a) ─► enc_v ─► z^v_t ─┐                                         │
  │ audio   o^a ─► F_a(a) ─► enc_a ─► z^a_t ─┤  dyn_s([z_t … z_{t-k+1}], action) ─► ẑ_{t+1}│
  │ touch   o^t ─► F_t(a) ─► enc_t ─► z^t_t ─┤  dec_s(ẑ_{t+1}) ─► ô_{t+1}              │
  │ proprio o^p ─► F_p(a) ─► enc_p ─► z^p_t ─┘                                         │
  └─────────────────────────────────────────────────────────────────────────────────────┘
                │ frozen latents z^s_t, modality mask m^s_t
                ▼
  ┌──────────── fusion world model (grown / re-trained per stage) ─────────────────────┐
  │ f_t = MLP( Σ_s m^s_t (P_s z^s_t + e_s) / Σ_s m^s_t )                              │
  │ state_t = dyn_f([f_t … f_{t-k+1}], action)        (predicts f_{t+1})              │
  │ read_s(f_t) ≈ z^s_t,  read_s(state_t) ≈ z^s_{t+1}  for every sense, even masked ones│
  └─────────────────────────────────────────────────────────────────────────────────────┘
```

* **F_s(a)** is a fixed, non-learned sensory front end whose parameters come from the
  capability tables (`dwm/capability.py`). It is the only place "age" enters the data.
* **enc/dec/dyn** are small and their widths scale with `w(a)`; the dynamics see the last
  `k(a)` latents (working memory) plus the infant's own motor command.
* **Growth** (`SenseWorldModel.grow`): new hidden units get random incoming weights and
  zero outgoing weights; new memory slots get zero input weights. The grown network
  computes exactly the same function as before (unit-tested in `tests/test_growth.py`),
  then continues learning with more capacity and better input.
* **Fusion** sees only latents, never raw input, so each sense model can be swapped,
  re-grown or ablated independently. Modality dropout teaches it to fill in a missing
  sense from the others.

### 2.1 Why late fusion of separately-grown models (and what we compare it to)

Three ways to combine:

1. **Early fusion** — one network over all raw senses. Simplest, strongest baseline,
   but cannot express sense-specific maturation and lets the easiest sense dominate.
2. **Late fusion of frozen sense models** (default). Matches the "sense-specific first,
   integrate later" story and makes every sense model reusable.
3. **Staged-onset fusion** — senses join the fusion model in the order they mature
   (touch/proprio → audio → vision), with fusion capacity growing too. This is the
   direct test of Turkewitz & Kenny's "limited input reduces inter-sensory competition".

The prototype implements (2). Phase 1 adds (1) as the baseline and (3) as the main
manipulation.

---

## 3. Per-sense specification

Values at the four prototype stages (full tables and interpolation in `dwm/capability.py`).

| Sense | Front-end parameters (0 → 3 → 6 → 12 mo) | Capacity | Core objective | Phase-1 data | Benchmarks |
|---|---|---|---|---|---|
| **Vision** | acuity 1 → 4 → 15 → 30 cpd; contrast sensitivity 2 → 10 → 50 → 100; colour saturation 0.1 → 0.7 → 1 → 1 | width ×1→×4, memory 1→6 steps | reconstruct + predict next frame, motion-weighted loss | SAYCam, BabyView (head-cam video) | object/face probes, VoE (permanence, solidity), BIB |
| **Audition** | frequency blur 2.5 → 0.5 bins; high-frequency pass 60% → 100%; binaural difference 0.3 → 1.0; noise floor 0.08 → 0.02 | same | predict next spectrogram frame | SAYCam audio, HomeBank day-long recordings | source localisation, phoneme discrimination and perceptual narrowing (Werker & Tees), word segmentation |
| **Touch** | receptive-field blur 1.2 → 0.3 taxels; noise 0.06 → 0.02 | same | predict next tactile frame given own action | MIMo full-body touch | shape-by-touch, contact prediction, self vs other touch |
| **Proprioception** (+ vestibular in Phase 2) | joint/hand noise 0.12 → 0.02 | same | forward model: next body state from motor command | MIMo | reaching accuracy, sensorimotor contingency detection |
| *Chemical senses* | — | — | — | — | Out of scope (no usable data or simulators); noted as a limitation. |

Each row is one independent model and one independently-reportable result, which is
what makes the per-sense framing useful for a paper: every sense gets its own
developmental trajectory figure.

---

## 4. Growth across ages

**Stages.** 0, 3, 6, 12 months in the prototype; 18 and 24 months once language
(word–referent grounding, à la Vong et al. 2024) is added.

**What grows at each stage.**

1. Sensory fidelity (the front-end filter relaxes along the capability tables).
2. Width of every hidden layer (×1, ×2, ×3, ×4).
3. Working-memory window (1, 2, 4, 6 steps; Phase 1 maps steps to seconds using the
   matrix in proposal §3.2).
4. **Nothing else.** The proposal adds object-tracking, agent and goal modules at 4, 6
   and 9 months; we do not. Anything gated by age would "emerge" on schedule by
   construction, which is circular. The capability matrix is the *evaluation target*:
   every capability has to be measured to emerge from the same objective (see the
   project's paper-framing note, `research/paper-framing.md`).

**Age = experience.** Each sense model is trained once, in age order, and an "age" is a
checkpoint after the matching amount of experience; the stages are checkpoints of one
continuous run, never separate models. Following the paper-framing note, the primary
maturation levers are *sensory immaturity* (implemented) and a *bottom-up plasticity
schedule* (Phase 1). Capacity growth is a factor we test, not an assumption: the brain
does not mature mainly by adding neurons (synapses overshoot then prune), so growing
networks are one level of the capacity factor alongside fixed-large, fixed-small and
over-provision-then-prune.

**Controls that make each lever a testable claim** (the four below are implemented for
each sense; Phase 1 adds shuffled-order and reversed-immaturity controls, ≥3 seeds per cell):

| Condition | Model size | Input fidelity | Purpose |
|---|---|---|---|
| **Grown** (developmental) | grows 0→12 | follows age | both levers together |
| **Adult-from-scratch** | 12-mo size throughout | adult throughout | standard ML baseline, same number of updates |
| **Curriculum-only** | 12-mo size throughout | follows age | sensory immaturity alone (the primary lever) |
| **Growth-only** | grows 0→12 | adult throughout | isolates the effect of capacity growth |

**Readouts.** Next-step prediction error on adult-quality input; linear probes for toy
position and identity; robustness of those probes to newborn-quality input (the
Vogelsang et al. 2018 prediction that early blur yields blur-robust representations);
violation-of-expectation index for permanence.

---

## 5. Hypotheses

* **H1 (trajectory).** Probe accuracy and VoE sensitivity in the grown models rise with
  age, and the *order* in which senses become informative follows their maturation
  order (touch/proprio, then audio, then vision).
* **H2 (starting small helps robustness).** Grown and curriculum-only vision models are
  more robust to degraded input than adult-from-scratch models, at little or no cost on
  clean input.
* **H3 (memory gates permanence).** A positive VoE index for occlusion appears only once
  the memory window spans the occlusion. Caveat: a window schedule is itself a capacity
  constraint we chose, so this tests "is memory span sufficient", not "does permanence
  emerge from experience alone"; the fixed-capacity conditions are the stricter test.
* **H7 (perceptual narrowing, flagship cross-sense test, Phase 1).** Trained on "native"
  input only, models first discriminate then lose non-native contrasts (speech
  phonemes, other-species faces) across checkpoints. Non-monotonic, cross-modal, and
  impossible to explain by "more data is better".
* **H4 (multisensory scaffolding).** A fused model tracks a toy through visual occlusion
  better than vision alone, because sound passes through occluders — so "object
  permanence" can appear earlier in a multisensory infant than in a vision-only one.
* **H5 (heterochrony).** Staged-onset fusion yields more balanced use of senses (less
  dominance by the easiest modality) than early fusion. *(Phase 1)*
* **H6 (brain alignment).** Age-matched models predict infant fMRI of that age better
  than mismatched ages (proposal Paper 2). *(Phase 4)*

---

## 6. Roadmap

| Phase | Scope | Concrete milestones | Compute |
|---|---|---|---|
| **0. Prototype** (done) | Synthetic multisensory nursery; 4 sense models; growth; fusion; VoE | `dwm/` package, unit tests, `scripts/run_experiments.py`, [RESULTS.md](RESULTS.md) | Laptop CPU, ~1 h |
| **1. Make it real** | Proper visual-angle optics and CSF; staged-onset fusion; early-fusion baseline; 3–5 seeds; IntPhys / ADEPT-style physics VoE; MNIST-like shape-by-touch | Paper-ready synthetic results for H1–H5 | 1 GPU-week |
| **2. Embodiment** | Port touch, proprioception and vestibular to MIMo (MuJoCo infant); babbling and reaching policies | Self-generated multisensory data at ages 0–12 mo; sensorimotor-contingency results | 1–2 GPU-weeks |
| **3. Naturalistic data** | Vision + audio from SAYCam/BabyView, age-binned by recording age; speech from HomeBank | Grown vs scratch on real infant-view data; phoneme narrowing curve | Several GPU-weeks; Databrary access |
| **4. Benchmarks & brains** | BIB (agents, goals), infant fMRI encoding models (dHCP, Kosakowski et al. data where shareable) | H6; proposal Paper 2 | Moderate |
| **5. Application** | The proposal's Streamlit "Infant Development Simulator", now backed by per-sense models (show "what the baby sees *and hears*") | Public demo | Small |

**Immediate next steps after this repo:** (1) 3+ seeds for every condition, (2) staged-onset
fusion and early-fusion baseline, (3) visual-angle optics, (4) write the arXiv preprint
around H1–H4 on synthetic data (Phase 0–1 results are enough for a workshop-style
preprint; Phase 2–3 for a main-conference paper).

---

## 7. arXiv paper plan

**Working title.** *Growing Senses: Sensory-Specific Developmental World Models of Infant Perception*

**Primary category** cs.LG; cross-list q-bio.NC and cs.AI. First-time submitters to a
category may need an endorsement from an existing arXiv author in that category.

**Outline (≈9 pages + appendix)**

1. Introduction — infants are not small adults; senses mature at different rates; the
   gap between developmental science and world-model ML.
2. Related work — world models (Ha & Schmidhuber; Dreamer; JEPA), developmental ML
   (Elman's starting small; Vogelsang et al. blur; Orhan et al. / Vong et al. on SAYCam;
   PLATO intuitive physics), embodied infant simulation (MIMo), network growth (Net2Net,
   progressive networks), multisensory development (Bahrick & Lickliter).
3. Method — capability tables per sense; sensory front ends; sense world models;
   function-preserving growth; fusion with modality dropout.
4. Experiments — (a) per-sense developmental trajectories; (b) grown vs scratch vs
   curriculum-only vs growth-only; (c) VoE and memory window; (d) fusion: occluded
   tracking and cross-modal fill-in.
5. Discussion — what transfers to real infant data; limits of synthetic worlds; the
   application.
6. Appendix — all age tables with sources, hyper-parameters, extra seeds.

A LaTeX skeleton with the prototype numbers filled in lives in [`paper/main.tex`](../paper/main.tex).

**Figures.** (1) system diagram; (2) what each sense delivers at 0/3/6/12 months
(rendered by `scripts/make_figures.py`); (3) trajectories of probe accuracy per sense
across age; (4) grown vs controls bar chart; (5) VoE index vs memory window; (6) fusion
vs vision-only occluded tracking.

---

## 8. Risks and how we handle them

| Risk | Mitigation |
|---|---|
| Synthetic worlds make results look cleaner than reality | Phase 3 repeats every comparison on SAYCam/BabyView; synthetic claims are framed as mechanism demonstrations |
| Capability tables are approximate and partly placeholder (audio, touch) | Every number lives in one file with a source note; sensitivity analysis over ±50% perturbations in Phase 1 |
| "Growth helps" may not hold at scale | All four control conditions are reported regardless of outcome |
| Data access (SAYCam via Databrary, infant fMRI) is slow | Start applications in Phase 0; synthetic + MIMo work proceeds in parallel |
| Over-claiming clinical use for the app | Keep the proposal's "educational, not diagnostic" framing |

---

## 9. References (verify every entry before submission)

1. Turkewitz, G., & Kenny, P. A. (1982). Limitations on input as a basis for neural organization and perceptual development. *Developmental Psychobiology*.
2. Elman, J. L. (1993). Learning and development in neural networks: the importance of starting small. *Cognition*.
3. Vogelsang, L., et al. (2018). Potential downside of high initial visual acuity. *PNAS*.
4. Bahrick, L. E., & Lickliter, R. (2000). Intersensory redundancy guides attentional selectivity and perceptual learning in infancy. *Developmental Psychology*.
5. Werker, J. F., & Tees, R. C. (1984). Cross-language speech perception. *Infant Behavior and Development*.
6. Spelke, E. S., & Kinzler, K. D. (2007). Core knowledge. *Developmental Science*.
7. Norcia, A. M., & Tyler, C. W. (1985). Spatial frequency sweep VEP. *Vision Research*.
8. Banks, M. S., & Bennett, P. J. (1988). Optical and photoreceptor immaturities limit the spatial and chromatic vision of human neonates. *JOSA A*.
9. Sullivan, J., et al. (2021). SAYCam. *Open Mind*.
10. Long, B., et al. (2024). The BabyView dataset. *arXiv*.
11. Orhan, A. E., Gupta, V. V., & Lake, B. M. (2020). Self-supervised learning through the eyes of a child. *NeurIPS*.
12. Vong, W. K., Wang, W., Orhan, A. E., & Lake, B. M. (2024). Grounded language acquisition through the eyes and ears of a single child. *Science*.
13. Gandhi, K., et al. (2021). Baby Intuitions Benchmark. *NeurIPS*.
14. Piloto, L. S., et al. (2022). Intuitive physics learning in a deep-learning model inspired by developmental psychology. *Nature Human Behaviour*.
15. Mattern, D., et al. (2023). MIMo: A multi-modal infant model. *arXiv:2312.04318*.
16. Ha, D., & Schmidhuber, J. (2018). World Models. *arXiv*.
17. Hafner, D., et al. (2023). Mastering diverse domains through world models (DreamerV3). *arXiv*.
18. LeCun, Y. (2022). A path towards autonomous machine intelligence. *OpenReview*.
19. Chen, T., Goodfellow, I., & Shlens, J. (2016). Net2Net. *ICLR*.
20. Rusu, A. A., et al. (2016). Progressive neural networks. *arXiv*.
21. Deen, B., et al. (2017). Organization of high-level visual cortex in human infants. *Nature Communications*.
22. Kosakowski, H. L., et al. (2022). Selective responses to faces, scenes, and bodies in the ventral visual pathway of infants. *Current Biology*.
