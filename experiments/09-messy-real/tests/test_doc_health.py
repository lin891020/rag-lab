"""09's doc_health adds garbage detection: a PDF whose embedded font has no ToUnicode map has a
text layer, but it reads as (cid:N) tokens. 08's checks say "text present"; 09 must say "garbage"."""

import json
from pathlib import Path

import pytest

from doc_health import garbage_share, inspect_pdf

HERE = Path(__file__).parents[1]
FIX = Path(__file__).parents[3] / "tests/fixtures"


def test_garbage_share_counts_cid_tokens_against_real_characters():
    assert garbage_share("第一條 本辦法依法訂定之。") == 0.0
    assert garbage_share("(cid:12)(cid:33)(cid:7)") == 1.0
    # 3 cid tokens next to 3 real characters: half garbage
    assert garbage_share("(cid:1)(cid:2)(cid:3)本辦法") == pytest.approx(0.5)
    assert garbage_share("") == 0.0


def test_clean_pdf_reports_no_garbage():
    h = inspect_pdf(FIX / "table56.pdf")
    assert h["garbage_pages"] == 0 and h["garbage_share"] == 0.0


def test_manifest_garbage_doc_is_flagged():
    """Once the corpus is built, the document the manifest marks as garbage must be flagged."""
    manifest = HERE / "corpus/manifest.json"
    if not manifest.exists():
        pytest.skip("corpus not built")
    docs = json.loads(manifest.read_text(encoding="utf-8"))["docs"]
    garbage = [d for d in docs if d["health"]["garbage_pages"] > 0]
    assert garbage, "the stratified sample must contain at least one garbage-text PDF"
    h = inspect_pdf(HERE / "corpus/pdf" / f"{garbage[0]['sysid']}.pdf")
    assert h["garbage_pages"] > 0 and h["garbage_share"] > 0.2


def test_rotated_page_is_counted(tmp_path):
    """A landscape 附表 page stored rotated: pdfplumber reads its characters backwards."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    tw = pymupdf.TextWriter(page.rect)
    tw.append(
        pymupdf.Point(50, 80),
        "修正規定 現行規定 說明 附表一 項次品目分類分級代碼",
        font=pymupdf.Font("cjk"),
        fontsize=10,
    )
    tw.write_text(page)
    page.set_rotation(90)
    pdf = tmp_path / "rot.pdf"
    doc.save(pdf)
    h = inspect_pdf(pdf)
    assert h["rotated_pages"] == 1
