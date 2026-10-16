---
title: "Internal referee report on the first draft (14 October 2026)"
subtitle: "Internal peer review, written as for a Nature-family submission, with the authors' response"
date: "15 October 2026"
---

# Summary of the submission

The manuscript builds one small predictive world model per sense (vision, hearing, touch, proprioception) in a synthetic multisensory scene, passes each sense through an age-indexed, non-learned "immaturity" filter, reads the models out at checkpoints labelled 0, 3, 6 and 12 months, and combines them with a fusion world model. Six training regimes separate sensory immaturity, capacity growth and data order. The main claims are: (i) the input schedule, not network growth, produces the infant-like trajectory; (ii) early immaturity buys robustness, and in hearing it improves clean performance; (iii) fusion carries a hidden object's position through occlusion better than any single sense; (iv) decodability of a hidden object does not imply surprise at its disappearance.

The question is important and timely, the code is open, and the authors are admirably careful not to over-claim. The framing — developmental constraints as hypotheses to test rather than modules to install — is the right one. In its current form, however, the evidence is too thin to support claims (i) and (ii) at the strength stated, and claim (iv) rests on one paradigm with a known confound. I list major concerns in order of importance, then minor ones.

# Major concerns

**M1. Ecological validity of the setting.** All results come from a 32×32 synthetic scene with one object of four types, a scripted hand and two-channel "audio" made of 32 frequency bins. The paper's title and abstract speak of infant perception. The authors should (a) state up front that this is a mechanism study in a controlled world, (b) say which result they expect to transfer to naturalistic data and why, and (c) give a concrete plan for the transfer (MIMo, SAYCam/BabyView). The broader question in the title should be answered in the terms the evidence supports.

**M2. The age effect may be an information effect.** The newborn vision filter at 32×32 leaves the toy barely visible. If the 0-month input contains almost no information about position, then a near-zero probe at 0 months says nothing about learning; it is guaranteed by the data-processing inequality. The authors must report an **information ceiling** per sense and age: the best a supervised model can do from the filtered input. Learning effects should then be stated relative to that ceiling. Relatedly, the hearing, touch and proprioception tables are placeholders; claims about the *order* of maturation across senses are therefore claims about the tables.

**M3. Probing on adult input conflates learning and input.** All trajectory probes are fitted and tested on adult-quality input. A model trained on blurred input is then evaluated out of distribution, which penalises the immature regimes at early checkpoints for reasons unrelated to what they have learned. The authors should add **own-age probes** (fitted and tested on the input quality the checkpoint was trained on) and report both.

**M4. Statistics.** Three seeds for the main regimes and one for the order controls; error bars of ±1 s.d.; no confidence intervals, no tests, no effect sizes. The robustness claim in vision is within seed noise by the authors' own admission. Required: more seeds for the controls, bootstrap confidence intervals, Welch tests and Hedges' g for each headline comparison, and a statement of which comparisons were planned.

**M5. The violation-of-expectation result is confounded.** In the "impossible" events the toy vanishes from every sense, so the fused model's positive VoE index can be (and, per the channel decomposition, is) explained by the sound stopping. Required: **dissociating events** — the toy vanishes visually but keeps sounding, and a silent toy — with per-channel surprise and paired effect sizes.

**M6. Fairness of the comparison.** Regimes are matched in updates but not in compute: grown models have fewer parameters for most of training. Hyperparameters were tuned (informally) on one regime. The authors should report parameter-update products per regime and state that hyperparameters were shared and not tuned per regime.

**M7. Linear probes measure decodability, not behaviour.** The trajectory figure is a figure of linear decodability. Infants are tested with looking-time paradigms. The paper partly concedes this in §4.4; it should be stated as a limitation of all probe-based results and the evaluation plan should make paradigms primary.

**M8. No theory.** The paper reports that vision models are "more surprised when the toy reappears than when it fails to" and that growth "slows learning," without explanation. Both have short explanations that the authors should give: (a) a model trained by mean-squared error predicts the conditional mean of the next frame; when an occluded object's re-emergence time is uncertain, the conditional mean is dominated by the occluder, so the error-minimising prediction is "nothing appears" and reappearance is the surprising event; (b) zero-initialised outgoing weights make the new units' gradients zero on the incoming side at the moment of growth, so new capacity contributes nothing until the outgoing weights move, which delays learning by construction. A brief formal treatment of the front ends (as operators), the objective, function-preserving growth, the probes and the indices belongs in Methods.

**M9. No quantitative link to infant data.** "Infant-like" is asserted from curve shapes. The authors should compute at least one alignment metric against published milestones (e.g. the age at which each sense reaches half its ceiling, and Kendall's τ between model and infant order) and be explicit that, with the current placeholder tables, agreement is expected by construction.

# Minor concerns

1. "Months" are stages of 1,500–2,000 updates. Use "checkpoint" or "stage" in figures and state the mapping in Methods.
2. Proprioception is at ceiling throughout and touch identity is low in every regime; say whether either sense can discriminate regimes at all, or drop them from the regime comparison.
3. Hearing identity is trivially 1.0 from pitch; the fused model "inherits" identity for free. Say so where fused identity is reported.
4. The VoE index is bounded in [−1, 1] and scale-free, but it hides the absolute surprise. Report $S_\text{possible}$ and $S_\text{impossible}$ alongside it, and a paired $d_z$.
5. The shuffled-order regime changes both order and variety; say which question it answers.
6. The companion studies use different code and stimuli. Present them as convergent evidence, in a separate figure, with their own n and caveats.
7. Figures: panel letters, consistent regime colours, units on every axis, n in every caption.
8. References: several are recent preprints; check every key before submission.

# Recommendation

Major revision. The direction is sound and the negative results (growth slows learning; decodable ≠ expected) are useful. The required analyses are cheap at this scale and should be done before any claim about "infant-like" learning is made.

# Authors' response and changes made

| # | Concern | What we did | Where |
|---|---|---|---|
| M1 | Ecological validity | Retitled around the question and its scoped answer; mechanism-study framing in the abstract; transfer plan kept in the roadmap | Title, Abstract, Discussion |
| M2 | Information ceiling | Supervised ceiling per sense at 8 ages; trajectories now shown against it | Fig. 2, Methods |
| M3 | Own-age probes | Own-age probes for every regime, seed and checkpoint (vision, hearing) | Fig. 2, Extended Data |
| M4 | Statistics | Two more seeds for both order controls (n = 3 everywhere); bootstrap 95% CIs, Welch tests and Hedges' g for every headline comparison | Fig. 3, Table 2 |
| M5 | VoE confound | Two dissociating events (visual-only vanish, silent toy) with per-channel surprise and $d_z$ | Fig. 4 |
| M6 | Compute | Parameter × update accounting per regime; shared hyperparameters stated | Methods, Table 3 |
| M7 | Probes ≠ behaviour | Stated as a limitation; paradigms made primary in the plan | Discussion |
| M8 | Theory | Box 1 and Methods: front ends as operators, conditional-mean account of the VoE failure, gradient account of growth delay, proof of function preservation | Box 1, Methods |
| M9 | Milestone alignment | Half-ceiling ages and Kendall τ against touch < hearing < vision, with the "by construction" caveat | Fig. 2, Results |
| m1–m8 | Minor | Addressed in the text and figures | — |

The outcomes of the new analyses, including those that weaken the original claims, are reported in the revised manuscript (`report/REPORT.md`).
