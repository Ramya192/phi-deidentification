"""
tests/test_document_ingestion.py

Covers ingestion/document_loader.py:
  1. .txt extraction, including latin-1 decode fallback (no deps needed).
  2. Unsupported file type raises UnsupportedFileTypeError.
  3. Empty/whitespace-only document raises EmptyDocumentError.
  4. .docx round-trip: build a real .docx in memory with python-docx,
     extract it back, verify the text (and a table cell) come through.
     Skipped if python-docx isn't installed.
  5. .pdf round-trip: build a minimal single-page PDF by hand (no
     reportlab/fpdf dependency needed just for a test fixture — the xref
     offsets are computed at runtime, not hardcoded), extract it with
     pdfplumber, verify the text comes through. Skipped if pdfplumber
     isn't installed.

Run with: pytest -v tests/test_document_ingestion.py
"""
from __future__ import annotations

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from ingestion.document_loader import (
    EmptyDocumentError,
    UnsupportedFileTypeError,
    load_text_from_bytes,
)


def test_txt_extraction_utf8():
    text = load_text_from_bytes(b"Patient John Doe, MRN 12345.", "note.txt")
    assert text == "Patient John Doe, MRN 12345."


def test_txt_extraction_latin1_fallback():
    # Not valid utf-8 on its own — should fall back to latin-1.
    raw = "café notes, temp 101.4°F".encode("latin-1")
    text = load_text_from_bytes(raw, "note.txt")
    assert "café" in text


def test_unsupported_file_type_raises():
    with pytest.raises(UnsupportedFileTypeError):
        load_text_from_bytes(b"whatever content", "chart.doc")


def test_empty_document_raises():
    with pytest.raises(EmptyDocumentError):
        load_text_from_bytes(b"   \n\n   \t  ", "empty.txt")


def test_docx_round_trip():
    docx = pytest.importorskip("docx")  # python-docx

    document = docx.Document()
    document.add_paragraph("Chief Complaint: follow-up visit.")
    document.add_paragraph("Patient Name: Jordan Ellis")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "MRN"
    table.rows[0].cells[1].text = "MR-778899"

    buf = io.BytesIO()
    document.save(buf)
    file_bytes = buf.getvalue()

    text = load_text_from_bytes(file_bytes, "note.docx")
    assert "Chief Complaint: follow-up visit." in text
    assert "Jordan Ellis" in text
    assert "MR-778899" in text  # table cell text should be included


def _build_minimal_pdf_bytes(text: str) -> bytes:
    """Hand-rolled minimal single-page PDF with real, computed xref offsets.
    Test-only — avoids needing reportlab/fpdf just to produce a fixture.
    Keep `text` free of parentheses/backslashes (not escaped here)."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> "
        b"/MediaBox [0 0 612 792] /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    stream_content = f"BT /F1 24 Tf 72 700 Td ({text}) Tj ET".encode("latin-1")
    objects.append(
        f"<< /Length {len(stream_content)} >>\nstream\n".encode("latin-1")
        + stream_content
        + b"\nendstream"
    )

    buf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(buf))
        buf += f"{i} 0 obj\n".encode("latin-1") + obj + b"\nendobj\n"

    xref_offset = len(buf)
    n = len(objects) + 1
    buf += f"xref\n0 {n}\n".encode("latin-1")
    buf += b"0000000000 65535 f \n"
    for off in offsets:
        buf += f"{off:010d} 00000 n \n".encode("latin-1")
    buf += f"trailer\n<< /Size {n} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode("latin-1")
    return bytes(buf)


def test_pdf_round_trip():
    pytest.importorskip("pdfplumber")

    pdf_bytes = _build_minimal_pdf_bytes("Hello PHI PDF Test")
    text = load_text_from_bytes(pdf_bytes, "note.pdf")
    assert "Hello PHI PDF Test" in text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
