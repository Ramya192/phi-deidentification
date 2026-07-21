"""
scripts/generate_demo_files.py

Converts a handful of existing .txt samples from data/txt_format/ into .pdf
and .docx versions, purely so you can manually try the POST /redact/upload
endpoint (and the Streamlit UI, once built) against real PDF/DOCX files
instead of only .txt.

These demo files are written to data/pdf_format/ and data/docx_format/ --
deliberately NOT added to ground_truth.json or used by eval/evaluate.py.
Converting to PDF/DOCX and back through pdfplumber/python-docx introduces
small whitespace/line-break differences from the original .txt, which would
silently invalidate the exact character-offset ground truth. Demo files
are for exercising the ingestion code path, not for scoring PHI-detection
accuracy -- data/txt_format/*.txt remains the only eval source.

All sample documents live under data/, grouped by file format:
data/txt_format/, data/pdf_format/, data/docx_format/.

USAGE
-----
    pip install fpdf2 python-docx
    python -m scripts.generate_demo_files
    python -m scripts.generate_demo_files --per-type 2   # more per type

OUTPUT
------
    data/pdf_format/<name>.pdf
    data/docx_format/<name>.docx
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.state import CLINICAL_DOC_TYPES

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
SOURCE_DIR = os.path.join(PROJECT_ROOT, "data", "txt_format")
PDF_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "pdf_format")
DOCX_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "docx_format")

# fpdf2's core fonts (Helvetica, Times, ...) only support latin-1, but the
# mtsamples source text contains "smart" typographic punctuation (curly
# apostrophes/quotes, en/em dashes, ellipses) that raises
# FPDFUnicodeEncodingException if passed through unmodified. Normalize the
# common cases to ASCII equivalents, then fall back to replacing anything
# still outside latin-1 so this can never crash on unusual input.
_UNICODE_TO_ASCII = {
    "‘": "'", "’": "'",   # smart single quotes
    "“": '"', "”": '"',   # smart double quotes
    "–": "-", "—": "--",  # en dash, em dash
    "…": "...",                # ellipsis
    " ": " ",                  # non-breaking space
}


def _sanitize_for_core_font(text: str) -> str:
    for unicode_char, ascii_equivalent in _UNICODE_TO_ASCII.items():
        text = text.replace(unicode_char, ascii_equivalent)
    # Safety net for anything else outside latin-1 (rare, but shouldn't crash the run).
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _pick_source_files(per_type: int) -> dict[str, list[str]]:
    """Picks up to `per_type` filenames per doc type from SOURCE_DIR,
    preferring the lowest-numbered (earliest-generated) files for
    determinism, so repeated runs pick the same demo set."""
    all_files = sorted(f for f in os.listdir(SOURCE_DIR) if f.endswith(".txt"))
    picks: dict[str, list[str]] = {dt: [] for dt in CLINICAL_DOC_TYPES}
    for filename in all_files:
        for dt in CLINICAL_DOC_TYPES:
            if filename.startswith(dt + "_") and len(picks[dt]) < per_type:
                picks[dt].append(filename)
                break
    return picks


def _write_docx(text: str, out_path: str) -> None:
    try:
        import docx  # python-docx
    except ImportError as exc:
        raise ImportError("DOCX generation requires: pip install python-docx") from exc

    document = docx.Document()
    for line in text.split("\n"):
        document.add_paragraph(line)
    document.save(out_path)


def _write_pdf(text: str, out_path: str) -> None:
    try:
        from fpdf import FPDF, XPos, YPos
    except ImportError as exc:
        raise ImportError("PDF generation requires: pip install fpdf2") from exc

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.set_auto_page_break(auto=True, margin=15)
    for line in text.split("\n"):
        line = _sanitize_for_core_font(line)
        # multi_cell wraps long lines automatically; blank lines still
        # advance the cursor so paragraph spacing is preserved.
        # new_x/new_y explicitly reset the cursor back to the left margin
        # after each call -- without this, fpdf2 leaves the cursor wherever
        # the previous line ended, so the *second* multi_cell call computes
        # almost no horizontal space left and raises FPDFException.
        pdf.multi_cell(0, 6, line if line.strip() else " ", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.output(out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-type", type=int, default=1, help="How many demo files per doc type (default: 1)")
    args = parser.parse_args()

    if not os.path.isdir(SOURCE_DIR):
        print(f"Source directory not found: {SOURCE_DIR}. Run "
              f"scripts.build_sample_dataset first.", file=sys.stderr)
        raise SystemExit(1)

    os.makedirs(PDF_OUTPUT_DIR, exist_ok=True)
    os.makedirs(DOCX_OUTPUT_DIR, exist_ok=True)
    picks = _pick_source_files(args.per_type)

    written = []
    for doc_type, filenames in picks.items():
        for filename in filenames:
            src_path = os.path.join(SOURCE_DIR, filename)
            with open(src_path, encoding="utf-8") as f:
                text = f.read()

            stem = os.path.splitext(filename)[0]
            docx_path = os.path.join(DOCX_OUTPUT_DIR, f"{stem}.docx")
            pdf_path = os.path.join(PDF_OUTPUT_DIR, f"{stem}.pdf")

            _write_docx(text, docx_path)
            _write_pdf(text, pdf_path)
            written.append((doc_type, docx_path, pdf_path))

    print(f"Generated {len(written)} doc-type samples as both .pdf ({PDF_OUTPUT_DIR}) "
          f"and .docx ({DOCX_OUTPUT_DIR}):\n")
    for doc_type, docx_path, pdf_path in written:
        print(f"  {doc_type}: {os.path.basename(docx_path)}, {os.path.basename(pdf_path)}")

    print(
        "\nTry one with the API:\n"
        f'  curl -X POST http://localhost:8000/redact/upload -F "file=@{written[0][2]}"'
        if written else ""
    )


if __name__ == "__main__":
    main()
