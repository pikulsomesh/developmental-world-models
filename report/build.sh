#!/usr/bin/env bash
# Rebuild the paper from results/:
#   1. fill every number in REPORT.template.md -> REPORT.md
#   2. LaTeX for Overleaf (natbib citations, refs.bib) -> report.tex
#   3. HTML with resolved references and MathML -> report.html, printed to report.pdf by headless Chromium
# Figures come from scripts/make_paper_figures.py (run it first).
set -e
cd "$(dirname "$0")"
cp ../paper/refs.bib refs.bib
python3 ../scripts/fill_report.py
pandoc REPORT.md -s --natbib -o report.tex
pandoc REPORT.md -s --citeproc --mathml --embed-resources --css paper.css --metadata link-citations=true \
  --metadata pagetitle="Do AI models learn like humans?" -o report.html
CHROME=${CHROME:-/opt/pw-browsers/chromium-1194/chrome-linux/chrome}
if [ -x "$CHROME" ]; then
  "$CHROME" --headless=new --no-sandbox --disable-gpu --no-pdf-header-footer --virtual-time-budget=10000 \
    --print-to-pdf="$PWD/report.pdf" "file://$PWD/report.html" 2>/dev/null
  mkdir -p ../docs/paper && cp report.pdf ../docs/paper/report.pdf
fi
echo built
