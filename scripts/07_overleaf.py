"""Bundle the paper for Overleaf. Reads paper/ and figures/, writes paper/overleaf.zip.

Upload the zip as a new Overleaf project (New Project > Upload Project) and compile
main.tex with pdfLaTeX. Rerun scripts 05 and 06 first so figures and numbers are current.
Run: .venv/bin/python scripts/07_overleaf.py
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
SOURCES = ["main.tex", "refs.bib", "numbers.tex", "table_results.tex"]


def main():
    out = PAPER / "overleaf.zip"
    figures = sorted((ROOT / "figures").glob("fig*.pdf"))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in SOURCES:
            bundle.write(PAPER / name, name)
        for figure in figures:
            bundle.write(figure, f"figures/{figure.name}")
    print(f"{out.relative_to(ROOT)}: {len(SOURCES)} sources, {len(figures)} figures")


if __name__ == "__main__":
    main()
