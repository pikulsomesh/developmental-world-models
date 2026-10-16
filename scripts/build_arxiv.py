"""Assemble the arXiv-ready LaTeX source (arxiv/main.tex) from report/REPORT.md.

Uses the Research Future template layout (Introduction, problem setting, theory, methods, results, discussion): article class, authblk, 1.2 spacing, no line numbers,
biblatex nejm numeric citations, sections Introduction / Materials and Methods / Results /
Discussion / back matter. Body text is converted from the (number-filled) Markdown report with
pandoc so wording and numbers always match the PDF and the site.

Usage: python scripts/build_arxiv.py   (run report/build.sh first, or via report/build.sh)
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess

ROOT = os.path.join(os.path.dirname(__file__), "..")
REP = os.path.join(ROOT, "report")
OUT = os.path.join(ROOT, "arxiv")
AUTHOR, AFFIL, EMAIL = "Somesh Mohapatra", "Independent", "pikulsomesh@gmail.com"
TITLE = "Do AI Models Learn Like Humans?"


def pandoc(md: str) -> str:
    r = subprocess.run(["pandoc", "-f", "markdown+tex_math_dollars", "-t", "latex", "--biblatex", "--ascii",
                        "--wrap=preserve"], input=md, capture_output=True, text=True, check=True)
    return r.stdout


def clean_md(md: str) -> str:
    md = re.sub(r"\*\*Fig\. (\d+) \| ", r"**", md)  # LaTeX adds "Figure N" itself
    md = re.sub(r"\*\*Extended Data Fig\. (\d+) \| ", "**", md)  # supplementary figures are numbered S1.. in LaTeX
    md = re.sub(r"^: Table \d+ \| ", ": ", md, flags=re.M)
    md = re.sub(r"\]\(figures/([^)]+)\)", r"](\1)", md)
    return md


UNICODE = {
    "\u00d7": r"\ensuremath{\times}", "\u2192": r"\ensuremath{\rightarrow}", "\u00b1": r"\ensuremath{\pm}",
    "\u2212": r"\ensuremath{-}", "\u00b2": r"\textsuperscript{2}", "\u03c4": r"\ensuremath{\tau}",
    "\u00b7": r"\textperiodcentered{}", "\u2248": r"\ensuremath{\approx}", "\u2264": r"\ensuremath{\leq}",
    "\u2265": r"\ensuremath{\geq}", "\u03b1": r"\ensuremath{\alpha}", "\u03c3": r"\ensuremath{\sigma}",
    "\u03bb": r"\ensuremath{\lambda}", "\u0394": r"\ensuremath{\Delta}", "\u2013": "--", "\u2014": "---",
    "\u00e9": r"\'e", "\u00f3": r"\'o", "\u2018": "`", "\u2019": "'", "\u201c": "``", "\u201d": "''",
    "\u2026": r"\ldots{}", "\u00a0": "~", "\u2208": r"\ensuremath{\in}", "\u221e": r"\ensuremath{\infty}",
    "\u2192": r"\ensuremath{\rightarrow}", "\u00b5": r"\ensuremath{\mu}", "\u03bc": r"\ensuremath{\mu}",
    "\u00b0": r"\textdegree{}", "\u00e4": r'\"a', "\u00f6": r'\"o', "\u00fc": r'\"u',
}


def ascii_tex(t: str) -> str:
    for k, v in UNICODE.items():
        t = t.replace(k, v)
    bad = sorted({c for c in t if ord(c) > 127})
    if bad:
        print("WARNING unmapped non-ASCII:", [(c, hex(ord(c))) for c in bad])
    return t


def number_equations(tex: str) -> str:
    return re.sub(r"\\\[(.*?)\\\]", lambda m: "\\begin{equation}" + m.group(1).strip() + "\\end{equation}", tex, flags=re.S)


def split_sections(md: str) -> dict:
    body = md.split("\n", 1)[1] if md.startswith("---") else md
    body = re.sub(r"^---\n.*?\n---\n", "", md, count=1, flags=re.S)
    parts = re.split(r"(?m)^# ", body)
    secs = {}
    for p in parts[1:]:
        title, _, rest = p.partition("\n")
        secs[title.strip()] = rest
    pre = parts[0]
    return pre, secs


def main():
    md = open(os.path.join(REP, "REPORT.md")).read()
    pre, secs = split_sections(md)
    abstract = re.search(r"::: summary\n(.+?)\n:::", pre, re.S).group(1).strip().strip("*").strip()
    for k, v in secs.items():  # drop any boxed perspectives; the theory lives in its own section
        m = re.search(r"::: box\n(.+?)\n:::", v, re.S)
        if m:
            secs[k] = v.replace(m.group(0), "")

    def conv(text, shift=0):
        t = pandoc(clean_md(text))
        t = number_equations(t)
        t = t.replace("\\section{", "\\section{").replace("\\hypertarget", "\\hypertarget")
        return t

    sec = lambda title, body: "\\section{" + title + "}\n\n" + body
    intro = sec("Introduction", conv(secs["Introduction"]))
    methods = sec("Materials and Methods", conv(secs["Materials and Methods"]))
    setting = sec("Problem setting", conv(secs["Problem setting"]))
    theory = sec("Theory", conv(secs["Theory"]))
    results = sec("Results", conv(secs["Results"]))
    disc = "\\section{Discussion}\n\n" + conv(secs["Discussion"])
    ed = ("\\clearpage\n\\section*{Extended Data (Supplementary Figures)}\n\\setcounter{figure}{0}\n"
          "\\renewcommand{\\thefigure}{S\\arabic{figure}}\n\n" + conv(secs["Extended Data"]))
    # headings: pandoc turns '##' into \subsection already when top-level is section? -> pandoc maps ## to \subsection with default top-level=section
    tpl = open(os.path.join(os.path.dirname(__file__), "arxiv_main.tex.in")).read()
    abstract_tex = pandoc(abstract).strip()
    refs_note = ""
    main_tex = (tpl.replace("%%TITLE%%", TITLE).replace("%%AUTHOR%%", AUTHOR).replace("%%AFFIL%%", AFFIL)
                .replace("%%EMAIL%%", EMAIL).replace("%%ABSTRACT%%", abstract_tex).replace("%%INTRO%%", intro + "\n" + setting + "\n" + theory)
                .replace("%%METHODS%%", methods).replace("%%RESULTS%%", results).replace("%%DISCUSSION%%", disc)
                .replace("%%BACKMATTER%%", conv(secs["Acknowledgments"]).replace("\\section{Acknowledgments}", "\\section*{Acknowledgments}")
                         + "\n" + conv(secs["Supplementary Materials"]).replace("\\section{Supplementary Materials}", "\\section*{Supplementary Materials}")
                         + "\n" + ed))
    main_tex = ascii_tex(main_tex)
    main_tex = re.sub(r"\\subsection\{(General|Author contributions|Funding|Competing interests|Data availability)\}", r"\\subsection*{\1}", main_tex)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    # Table 1 has a long first column: give it more room than the four value columns
    main_tex = main_tex.replace("(\\columnwidth - 8\\tabcolsep) * \\real{0.2000}", "(\\columnwidth - 8\\tabcolsep) * \\real{0.1800}", 4).replace("(\\columnwidth - 8\\tabcolsep) * \\real{0.1800}", "(\\columnwidth - 8\\tabcolsep) * \\real{0.2800}", 1)
    # keep figure aspect ratio; pandoc's width+height pair would stretch the image
    main_tex = main_tex.replace("width=1\\textwidth,height=\\textheight", "width=\\textwidth,height=0.78\\textheight,keepaspectratio")
    main_tex = main_tex.replace("\\begin{figure}\n", "\\begin{figure}[tbp]\n")
    open(os.path.join(OUT, "main.tex"), "w").write(main_tex)
    shutil.copy(os.path.join(REP, "refs.bib"), os.path.join(OUT, "references.bib"))
    for f in os.listdir(os.path.join(REP, "figures")):
        if re.match(r"(fig\d|ed\d).*\.png$", f):
            shutil.copy(os.path.join(REP, "figures", f), os.path.join(OUT, "figures", f))
    print("wrote arxiv/main.tex")


if __name__ == "__main__":
    main()
