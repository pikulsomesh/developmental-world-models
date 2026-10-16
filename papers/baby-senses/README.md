# Paper 2: What Infants Sense at Each Age

*What Infants Sense at Each Age: Age-Tuned Neural Networks and an Interactive Simulator* (draft).

A developmental-neuroscience paper on **sensory perception, not cognition**: what infants perceive through touch, taste and smell, hearing and vision from birth to 24 months; how age-indexed front ends plus small prediction networks give a quick simulation of it; and the Baby Senses Lab (`docs/app/`), a browser-only simulation and research tool (not a screening or diagnostic test).

- `main.tex`, `references.bib`, `main.bbl`: LaTeX source (Research Future template layout, biblatex NEJM style). Compile with `pdflatex main && bibtex main && pdflatex main && pdflatex main`. Include `main.bbl` for arXiv.
- `baby-senses.pdf`: built PDF (also served at `docs/paper/baby-senses.pdf`).
- `figures/`: Figs. 1-5. `code/make_figures.py` draws them from the app's age tables (`docs/app/lab-data.js`), `results/extra.json` and the companion studies' raw results (`papers/world-models/companion/`); Fig. 5 is cropped from screenshots of the live site.

**Relation to paper 1** (`papers/world-models/`): the two papers share the vision front-end values and equations and the main-study training runs and information ceilings behind Fig. 3; each paper states this where it occurs. No figure is reproduced between the papers.
