# Paper 1, cs.LG draft: what changed

Draft only. The submitted source in `../arxiv/` and `arxiv_packages/paper1/` is untouched. Figures are read from `../arxiv/figures/`; no figure, number or experiment changed.

## Title
Input Constraints and Learning Dynamics in Predictive World Models

Alternatives:
1. Contrast Sets the Speed, Blur Sets the Ceiling: Learning Dynamics of Predictive World Models under Degraded Input
2. Learning Dynamics of Predictive World Models under Degraded, Staged and Multimodal Input

## Structure (old -> new)
- Abstract: leads with the ML problem and the four propositions, then the architecture, then results; ends with the scope limit. Plain-ASCII copy in `abstract_plain_ascii.txt` (1,886 characters; arXiv limit 1,920).
- Introduction: world models and data limits first, then curriculum learning, then the four propositions in prose. The biological schedule is now one short paragraph ("Motivation from biological learners"). The companion-paper overlap note is kept, shortened.
- New Section 2, "Problem setting and model": the problem statement (was the start of Theory), the vision degradation operator equations and Table 1 (were in Methods), the per-modality world model with the spatial-broadcast decoder and the loss (were in Methods), the fusion model with modality masking (was in Methods), the curricula table and ablations (were in Theory and Methods), and the evaluation protocol (was split across Theory and Methods).
- Section 3, Theory: same four propositions, statements and proofs unchanged in substance, now in theorem environments. One sentence added that relates the zero-outgoing-weight widening to Net2Net.
- Section 4, Results: same results and numbers, reordered so the spectral and Jacobian findings follow the ablations and the experience-matched check comes last.
- Discussion: same content and caveats.
- Appendix: environment, non-visual operators, training, spectral analysis, read-outs, follow-up experiments, statistics, and the half-ceiling maturation-order result (moved out of the main Results).

## Wording
- infant, baby, newborn -> biological learners, stage a = 0, degraded input; "Immature input" -> degraded-to-clean curriculum; "Adult input" -> clean baseline; regimes -> curricula; hearing -> audio; toy -> object; babbling hand -> action-driven effector.
- Stages are indexed by a in {0, 3, 6, 12}, the age in months whose parameters set the operator, because the figures are labelled in months.

## References added (each checked against its source)
- Bengio et al. 2009, Curriculum Learning, ICML, doi 10.1145/1553374.1553380
- Chen, Goodfellow, Shlens 2016, Net2Net, ICLR, arXiv 1511.05641
- Kingma and Ba 2015, Adam, ICLR, arXiv 1412.6980
- Neverova et al. 2016, ModDrop, IEEE TPAMI, arXiv 1501.00102 (volume and pages not confirmed, so left out)

## Not changed
- Figure images still contain the words "Synthetic nursery" (Fig. 1a), "infant order" (Fig. 3f), "Immature input" and "Adult input" (legends), and month axes. The captions map them to the new terms.
