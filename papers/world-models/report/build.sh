#!/usr/bin/env bash
# Rebuild the paper from results/:
#   1. fill every number in REPORT.template.md -> REPORT.md
#   2. arXiv-format LaTeX source (Research Future template layout) -> ../arxiv/
#   3. HTML with resolved references and MathML -> report.html, printed to report.pdf by headless Chromium
# Figures come from scripts/make_paper_figures.py (run it first).
set -e
cd "$(dirname "$0")"
python3 ../../../scripts/fill_report.py
python3 ../../../scripts/build_arxiv.py
pandoc REPORT.md -s --citeproc --bibliography=refs.bib --katex=file://$PWD/vendor/katex/ --embed-resources --css paper.css --metadata link-citations=true \
  --metadata pagetitle="Do AI Models Learn Like Humans?" -o report.html
CHROME=${CHROME:-/opt/pw-browsers/chromium-1194/chrome-linux/chrome}
if [ -x "$CHROME" ]; then
  "$CHROME" --headless=new --no-sandbox --disable-gpu --no-pdf-header-footer --virtual-time-budget=10000 \
    --print-to-pdf="$PWD/report.pdf" "file://$PWD/report.html" 2>/dev/null
  mkdir -p ../../../docs/paper && cp report.pdf ../../../docs/paper/report.pdf
fi
echo built
