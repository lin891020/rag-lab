"""The ocrrows route = OCR pages swapped in first, then the comparison-table pages replaced by
records. Synthetic PDFs for the mechanics; 09's corpus for the no-overlap claim."""

import json
from pathlib import Path

import pymupdf
import pytest

from extract import CORPUS09, ocrrows_text, rows_text

W, H = 595, 842
FONT = pymupdf.Font("cjk")


def _plain_pdf(tmp_path: Path, lines: list[str]) -> Path:
    doc = pymupdf.open()
    for ln in lines:
        p = doc.new_page(width=W, height=H)
        tw = pymupdf.TextWriter(p.rect)
        tw.append(pymupdf.Point(50, 80), ln, font=FONT, fontsize=10)
        tw.write_text(p)
    out = tmp_path / "plain.pdf"
    doc.save(out)
    return out


def test_rows_text_takes_the_pages_it_is_given(tmp_path):
    pdf = _plain_pdf(tmp_path, ["第一頁的字夠長到算有文字層了吧", "第二頁的字也夠長到算有文字層"])
    assert rows_text(pdf, pages=["甲", "乙"]) == "甲\n乙"  # no table -> pages pass through


def test_ocrrows_equals_rows_when_nothing_needs_ocr(tmp_path):
    pdf = _plain_pdf(tmp_path, ["第一頁的字夠長到算有文字層了吧", "第二頁的字也夠長到算有文字層"])
    assert ocrrows_text(pdf, tmp_path / "cache") == rows_text(pdf)


@pytest.mark.skipif(not (CORPUS09 / "extract_stats.json").exists(), reason="09 corpus missing")
def test_ocrrows_files_match_ocr_on_scanned_docs_and_rows_on_table_docs():
    here = Path(__file__).parents[1] / "corpus" / "text"
    if not list(here.glob("*.ocrrows.txt")):
        pytest.skip("run extract.py first")
    stats = json.loads((CORPUS09 / "extract_stats.json").read_text("utf-8"))
    for sysid, s in stats.items():
        got = (here / f"{sysid}.ocrrows.txt").read_text("utf-8")
        if s["ocr_pages"]:
            assert s["records"] == 0  # the two kinds of doc do not overlap
            assert got == (CORPUS09 / "text" / f"{sysid}.ocr.txt").read_text("utf-8")
        else:
            assert got == (CORPUS09 / "text" / f"{sysid}.rows.txt").read_text("utf-8")
