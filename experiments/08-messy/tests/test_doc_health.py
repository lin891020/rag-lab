"""doc_health.py must flag each of 08's messes on the PDFs degrade.py makes."""

from pathlib import Path

import pymupdf
from PIL import Image

from degrade import borderless_table, header_footer, two_column, typeset
from doc_health import inspect_pdf

FIX = Path(__file__).parents[3] / "tests/fixtures"


def test_clean_ruled_table_pdf(tmp_path):
    h = inspect_pdf(FIX / "table56.pdf")
    assert h["pages"] == 2 and h["text_pages"] == 2
    assert h["table_pages"] == 2 and h["repeated_lines"] == 0 and h["two_column_pages"] == 0


def test_header_footer_is_reported_as_repeated_lines(tmp_path):
    out = tmp_path / "hf.pdf"
    header_footer(FIX / "table587-3.pdf", out, title="用戶用電設備裝置規則")
    h = inspect_pdf(out)
    assert h["repeated_lines"] >= 1 and "用戶用電設備裝置規則" in h["repeated_examples"]


def test_two_column_pages_are_detected(tmp_path):
    out = tmp_path / "two.pdf"
    items = [("甲" * 30, "1")] * 300
    two_column(items, out)
    h = inspect_pdf(out)
    # the last page may hold only a half-filled left column
    assert h["pages"] >= 2 and h["two_column_pages"] >= h["pages"] - 1


def test_borderless_table_is_suspected(tmp_path):
    out = tmp_path / "t.pdf"
    t = {
        "title": "表九九",
        "names": ["線徑", "安培容量", "備註"],
        "rows": [[str(i), str(i * 3), "x"] for i in range(1, 15)],
        "notes": "",
    }
    borderless_table(t, out)
    h = inspect_pdf(out)
    assert h["table_pages"] == 0 and h["aligned_text_pages"] == 1


def test_image_only_pdf_has_no_text_layer(tmp_path):
    png = tmp_path / "p.png"
    Image.new("L", (600, 800), 255).save(png)
    pdf = tmp_path / "scan.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, filename=str(png))
    doc.save(pdf)
    h = inspect_pdf(pdf)
    assert h["pages"] == 1 and h["text_pages"] == 0


def test_prose_with_article_headers_is_not_a_table_nor_two_column_nor_repeated(tmp_path):
    out = tmp_path / "prose.pdf"
    items = []
    for n in range(1, 120):
        items.append((f"第 {n} 條", str(n)))
        items.append(("甲" * 45, str(n)))
        items.append(("一、" + "乙" * 30, str(n)))
    typeset(items, out, columns=1)
    h = inspect_pdf(out)
    assert h["pages"] >= 3 and h["aligned_text_pages"] == 0 and h["two_column_pages"] == 0
    assert h["repeated_lines"] == 0
