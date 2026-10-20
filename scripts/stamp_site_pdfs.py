"""Stamp every link to a site PDF with a short hash of the file, so browsers and the
Pages CDN fetch the new copy whenever a paper is rebuilt. Run after copying a PDF into docs/paper/."""
import hashlib, pathlib, re

DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"
for pdf in sorted((DOCS / "paper").glob("*.pdf")):
    v = hashlib.md5(pdf.read_bytes()).hexdigest()[:8]
    pat = re.compile(r"paper/" + re.escape(pdf.name) + r"(\?v=[0-9a-f]+)?")
    for f in list(DOCS.rglob("*.html")) + list(DOCS.rglob("*.js")):
        s = f.read_text()
        t = pat.sub(f"paper/{pdf.name}?v={v}", s)
        if t != s:
            f.write_text(t)
    print(f"{pdf.name}: v={v}")
