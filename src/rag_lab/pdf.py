"""PDF to text. Two shapes this lab meets: running text and one-table attachments.

Both go through plain text extraction. pdfplumber's grid extraction was tried on the 109
table attachments of 用戶用電設備裝置規則 and folded the rows of 55 of them into single cells
(no ruling lines between rows); the text layer keeps one row per line, so that is what we keep.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

logger = logging.getLogger(__name__)


def extract_text(path: Path | str) -> str:
    """Page texts joined with newlines. No layout analysis; good enough for 法規 text."""
    pages = [p.extract_text() or "" for p in PdfReader(str(path)).pages]
    logger.debug("pdf text", extra={"ctx": {"path": str(path), "pages": len(pages)}})
    return "\n".join(pages)


def table_to_text(path: Path | str) -> str:
    """Title line first, then the page text as pdfplumber lays it out (one table row per line).

    Some attachments draw a formula above the title; the title is the first line starting with 表.
    """
    with pdfplumber.open(str(path)) as pdf:
        lines = [
            ln.strip() for page in pdf.pages for ln in (page.extract_text() or "").splitlines()
        ]
    lines = [ln for ln in lines if ln]
    title_at = next((i for i, ln in enumerate(lines) if ln.startswith("表")), 0)
    body = [lines[title_at]] + lines[:title_at] + lines[title_at + 1 :]
    logger.debug("table text", extra={"ctx": {"path": str(path), "lines": len(body)}})
    return "\n".join(body)
