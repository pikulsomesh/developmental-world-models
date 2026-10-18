# Paper 1: Learning Like Infants: Input Constraints in Predictive World Models

- `report/REPORT.template.md`: source with `{{...}}` placeholders filled from `results/` by `scripts/fill_report.py`
- `report/report.pdf`: built paper (`report/build.sh`); `report/REFERENCES_VERIFIED.md`: citation audit; `report/PEER_REVIEW.md`: internal referee report
- `arxiv/`: arXiv-format LaTeX (`main.tex`, `references.bib`, `figures/`); see `arxiv/README.md`
- `companion/`: code and results of the companion experiments

Rebuild: `python scripts/analyze_spectral.py && python scripts/make_paper_figures.py && papers/world-models/report/build.sh`
