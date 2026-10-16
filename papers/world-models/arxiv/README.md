# arXiv submission source

Built on the Research Future template layout (`authblk`, 1.2 spacing, no line numbers, `biblatex` with the NEJM numeric style).

- `main.tex`: manuscript (generated from `report/REPORT.template.md` by `scripts/build_arxiv.py`; edit the template, not this file)
- `references.bib`: 30 references, each checked against a primary record; see `report/REFERENCES_VERIFIED.md`
- `figures/`: Figures 1-5 and Extended Data Figures S1-S3 (PNG)

## Compile

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Or upload the folder to Overleaf (compiler: pdfLaTeX). For the arXiv upload include the generated `main.bbl`, since arXiv does not run bibtex for biblatex sources.

## Notes

- The correspondence address in `main.tex` is the account email; edit `EMAIL` in `scripts/build_arxiv.py` to change it.
- This source was generated without a TeX engine in the build environment (linted with chktex only), so compile it once on Overleaf and fix any warnings before submitting.
