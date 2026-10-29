"""The three extraction routes on synthetic gazette-shaped PDFs (built with PyMuPDF, so the
tests run on either machine without the corpus)."""

from pathlib import Path

import pymupdf
import pytest

from extract import compare_rows, law_name, ocr_pages, records_to_text


def _n(s: str) -> str:
    return s.replace("\n", "")

FONT = pymupdf.Font("cjk")
W, H = 595, 842


def _draw_table(
    page: pymupdf.Page, rows: list[list[str]], col_x: list[float], y0: float, row_h: float
) -> None:
    """A ruled table: col_x are the vertical rule positions (len = cols + 1)."""
    tw = pymupdf.TextWriter(page.rect)
    y = y0
    for row in rows:
        for j, cell in enumerate(row):
            for k, ln in enumerate(cell.split("\n")):
                tw.append(pymupdf.Point(col_x[j] + 4, y + 14 + 14 * k), ln, font=FONT, fontsize=10)
        page.draw_rect(pymupdf.Rect(col_x[0], y, col_x[-1], y + row_h), width=0.5)
        for x in col_x[1:-1]:
            page.draw_line(pymupdf.Point(x, y), pymupdf.Point(x, y + row_h), width=0.5)
        y += row_h
    tw.write_text(page)


def _footer(page: pymupdf.Page, n: int) -> None:
    tw = pymupdf.TextWriter(page.rect)
    tw.append(pymupdf.Point(W / 2, H - 20), str(n), font=FONT, fontsize=9)
    tw.write_text(page)


@pytest.fixture
def two_page_table(tmp_path: Path) -> Path:
    """Header row on both pages, 第一條's amended cell cut by the page break, 第二條 on page 2,
    and a page number under each table (the footer that defeats 05's continuation rule)."""
    doc = pymupdf.open()
    cols = [50.0, 230.0, 410.0, 560.0]
    header = ["修 正 條 文", "現 行 條 文", "說 明"]
    p1 = doc.new_page(width=W, height=H)
    _draw_table(
        p1,
        [
            header,
            [
                "第一條 本辦法依法訂定\n之，補助上限為",
                "第一條 本辦法依法訂定\n之。",
                "新增補助上限。",
            ],
        ],
        cols,
        60,
        60,
    )
    _footer(p1, 1)
    p2 = doc.new_page(width=W, height=H)
    _draw_table(
        p2,
        [
            header,
            ["新臺幣三萬元。", "", ""],
            [
                "第二條 本辦法自發布日\n施行。",
                "第二條 本辦法自中華民\n國一百年施行。",
                "改為發布日。",
            ],
        ],
        cols,
        60,
        60,
    )
    _footer(p2, 2)
    out = tmp_path / "draft.pdf"
    doc.save(out)
    return out


def test_compare_rows_joins_the_cut_cell_and_drops_header_rows(two_page_table):
    recs = compare_rows(two_page_table)
    assert [r["label"] for r in recs] == ["1", "2"]
    assert _n(recs[0]["amended"]) == "本辦法依法訂定之，補助上限為新臺幣三萬元。"
    assert _n(recs[0]["current"]) == "本辦法依法訂定之。"
    assert recs[0]["note"] == "新增補助上限。"
    assert _n(recs[1]["amended"]) == "本辦法自發布日施行。"
    assert "\n" in recs[0]["amended"]  # cell lines are kept, so long records can be split
    assert recs[1]["pages"] == [1]
    assert recs[0]["pages"] == [0, 1]


def test_compare_rows_maps_a_split_column_back_to_three(tmp_path):
    """A page where the 說明 column is drawn as two sub-columns (5 rules instead of 4)."""
    doc = pymupdf.open()
    p = doc.new_page(width=W, height=H)
    _draw_table(
        p,
        [["修正條文", "現行條文", "說明", ""], ["第三條 甲。", "第三條 乙。", "一、", "理由。"]],
        [50.0, 230.0, 410.0, 480.0, 560.0],
        60,
        40,
    )
    out = tmp_path / "split.pdf"
    doc.save(out)
    recs = compare_rows(out)
    assert len(recs) == 1 and recs[0]["label"] == "3"
    assert _n(recs[0]["note"]) == "一、理由。"


def test_records_to_text_is_parseable_by_the_cn_parser():
    from chunkers import parse_articles_cn

    recs = [
        {"label": "1", "amended": "甲。", "current": "乙。", "note": "改。", "pages": [0]},
        {"label": "14-1", "amended": "丙。", "current": "", "note": "新增。", "pages": [1]},
    ]
    text = records_to_text(recs)
    arts = parse_articles_cn(text)
    assert [a.label for a in arts] == ["1", "14-1"]
    body = arts[0].body.replace("\n", "")
    assert "修正條文：甲。" in body and "現行條文：乙。" in body and "說明：改。" in body


def test_law_name_takes_the_first_quoted_name():
    assert law_name("農業部公告：預告「漁船船員管理規則」部分條文修正草案") == "漁船船員管理規則"
    assert law_name("修正「嘉義市A辦法」，名稱並修正為「嘉義市B辦法」") == "嘉義市A辦法"
    assert law_name("增訂不動產估價師法第十八條之一條文") == "不動產估價師法"
    assert law_name("修正中小企業發展條例第一條、第二條條文") == "中小企業發展條例"


def test_ocr_pages_picks_textless_and_garbage_pages(tmp_path):
    doc = pymupdf.open()
    good = doc.new_page(width=W, height=H)
    tw = pymupdf.TextWriter(good.rect)
    tw.append(
        pymupdf.Point(50, 80),
        "第一條 本辦法依法訂定之，內容足夠長到算有文字。",
        font=FONT,
        fontsize=10,
    )
    tw.write_text(good)
    doc.new_page(width=W, height=H)  # blank = scanned-looking page
    out = tmp_path / "mix.pdf"
    doc.save(out)
    assert ocr_pages(out) == [1]


def test_compare_rows_skips_pages_pdfplumber_reads_backwards(tmp_path):
    """A rotated page is left to the base route: no record is built from reversed text."""
    doc = pymupdf.open()
    p = doc.new_page(width=W, height=H)
    _draw_table(
        p,
        [["修正條文", "現行條文", "說明"], ["第三條 甲。", "第三條 乙。", "理由。"]],
        [50.0, 230.0, 410.0, 560.0],
        60,
        40,
    )
    p.set_rotation(90)
    out = tmp_path / "rot.pdf"
    doc.save(out)
    assert compare_rows(out) == []
