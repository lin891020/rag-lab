"""Each degradation must produce the failure it is meant to simulate, measured with the same
extractors the experiment uses (pypdf stream order, pdfplumber y-order, pdfplumber lines)."""

from pathlib import Path

import pdfplumber
from pypdf import PdfReader

from degrade import (
    borderless_table,
    concat,
    degrade_png,
    header_footer,
    two_column,
)

FIX = Path(__file__).parents[3] / "tests/fixtures"
LAW = Path(__file__).parents[2] / "01-chunking/corpus/law.pdf"


def test_header_footer_lands_on_every_page(tmp_path):
    out = tmp_path / "hf.pdf"
    info = header_footer(FIX / "table56.pdf", out, title="用戶用電設備裝置規則")
    pages = [p.extract_text() or "" for p in PdfReader(str(out)).pages]
    assert info["pages"] == len(pages) == 2
    assert all("用戶用電設備裝置規則" in t for t in pages)
    assert "第 2 頁／共 2 頁" in pages[1]
    # the original content is still there
    assert "表五六" in pages[0]


def test_two_column_interleaves_under_pdfplumber_but_not_pypdf(tmp_path):
    items = [(f"第 {n} 條", str(n)) for n in range(1, 3)]
    items = [
        ("第 1 條", "1"),
        ("甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲", "1"),
    ] * 60 + [
        ("第 2 條", "2"),
        ("乙乙乙乙乙乙乙乙乙乙乙乙乙乙乙乙乙乙乙乙乙乙乙乙乙乙乙乙乙乙乙乙乙乙乙乙乙乙乙乙", "2"),
    ] * 60
    out = tmp_path / "two.pdf"
    typeset = two_column(items, out)
    # every typeset line keeps its label and long lines were wrapped
    assert all(lbl in ("1", "2") for _, lbl in typeset)
    assert max(len(t) for t, _ in typeset) < 40
    assert len(typeset) > len(items)
    # pypdf (content-stream order) keeps columns apart
    stream = "\n".join(p.extract_text() or "" for p in PdfReader(str(out)).pages)
    assert "第 1 條" in stream.splitlines()
    # pdfplumber (y-sorted) glues the two columns into one line somewhere
    with pdfplumber.open(out) as pdf:
        plumber = "\n".join(p.extract_text() or "" for p in pdf.pages)
    assert any("甲" in ln and "乙" in ln for ln in plumber.splitlines())


def test_concat_appends_pages(tmp_path):
    out = tmp_path / "cat.pdf"
    info = concat(FIX / "table56.pdf", FIX / "table587-3.pdf", out)
    assert info["pages"] == len(PdfReader(str(out)).pages) == 2 + 3


def test_degrade_png_levels(tmp_path):
    from PIL import Image

    src = tmp_path / "src.png"
    Image.new("L", (600, 800), 255).save(src)
    phone = degrade_png(src, tmp_path / "phone.png", level="phone", seed=0)
    copy3 = degrade_png(src, tmp_path / "copy3.png", level="copy3", seed=0)
    assert phone["rotate_deg"] != 0 and phone["dpi_scale"] == 200 / 300
    assert copy3["dpi_scale"] == 150 / 300 and copy3["salt_pepper"] > 0
    im = Image.open(tmp_path / "phone.png")
    assert im.size == (400, 533)
    im3 = Image.open(tmp_path / "copy3.png")
    assert set(im3.convert("L").tobytes()) <= {0, 255}


def test_borderless_table_has_text_but_no_ruled_table(tmp_path):
    t = {
        "title": "表九九",
        "names": ["線徑", "安培容量"],
        "rows": [["2.0", "15"], ["5.5", "30"]],
        "notes": "註：測試",
    }
    out = tmp_path / "t.pdf"
    borderless_table(t, out)
    with pdfplumber.open(out) as pdf:
        page = pdf.pages[0]
        assert page.find_tables() == []  # lines strategy: nothing to find
        text = page.extract_text() or ""
        assert "表九九" in text and "5.5" in text and "註：測試" in text
        found = page.find_tables({"vertical_strategy": "text", "horizontal_strategy": "text"})
        assert found and len(found[0].extract()) >= 3
