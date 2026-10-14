#!/usr/bin/env bash
# Rebuild the report: copy current figures, then make a LaTeX version for Overleaf
# (natbib citations, refs.bib) and a self-contained HTML preview with resolved references.
set -e
cd "$(dirname "$0")"
cp ../results/figures/fig_*.png figures/
cp ../paper/refs.bib refs.bib
pandoc REPORT.md -s --natbib -o report.tex
pandoc REPORT.md -s --citeproc --embed-resources --metadata link-citations=true -o report.html
echo built
