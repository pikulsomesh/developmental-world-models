#!/usr/bin/env bash
# Rebuild the paper from results/:
#   1. fill every number in REPORT.template.md -> REPORT.md
#   2. arXiv-format LaTeX source (Research Future template layout) -> ../arxiv/
#   3. HTML with resolved references (report.html) and the PDF compiled from the LaTeX source
# Figures come from scripts/make_paper_figures.py (run it first).
set -e
cd "$(dirname "$0")"
python3 ../../../scripts/fill_report.py
python3 ../../../scripts/build_arxiv.py
pandoc REPORT.md -s --citeproc --bibliography=refs.bib --katex=file://$PWD/vendor/katex/ --embed-resources --css paper.css --metadata link-citations=true \
  --metadata pagetitle="Learning Like Infants: Input Constraints in Predictive World Models" -o report.html
# Primary PDF: the LaTeX source compiled with pdfLaTeX + BibTeX (same form as the companion paper)
if command -v pdflatex >/dev/null; then
  (cd ../arxiv && pdflatex -interaction=nonstopmode main >/dev/null && bibtex main >/dev/null; \
   pdflatex -interaction=nonstopmode main >/dev/null && pdflatex -interaction=nonstopmode main >/dev/null)
  cp ../arxiv/main.pdf report.pdf
  mkdir -p ../../../docs/paper && cp report.pdf ../../../docs/paper/report.pdf
  python3 ../../../scripts/stamp_site_pdfs.py
fi
echo built
