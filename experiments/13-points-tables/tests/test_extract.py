"""The rows route: a ruled table on a page becomes one line per data row, each cell written as
表頭：值 (03's key-value shape); pages without a table keep their text."""

from pathlib import Path

import pymupdf

from extract import ROUTES, rows_lines, rows_text

W, H = 595, 842
FONT = pymupdf.Font("cjk")


def _draw_table(page, rows, col_x, y0, row_h):
    tw = pymupdf.TextWriter(page.rect)
    y = y0
    for row in rows:
        for j, cell in enumerate(row):
            tw.append(pymupdf.Point(col_x[j] + 4, y + 14), cell, font=FONT, fontsize=10)
        page.draw_rect(pymupdf.Rect(col_x[0], y, col_x[-1], y + row_h), width=0.5)
        for x in col_x[1:-1]:
            page.draw_line(pymupdf.Point(x, y), pymupdf.Point(x, y + row_h), width=0.5)
        y += row_h
    tw.write_text(page)


def _pdf(tmp_path: Path) -> Path:
    doc = pymupdf.open()
    p1 = doc.new_page(width=W, height=H)
    tw = pymupdf.TextWriter(p1.rect)
    tw.append(
        pymupdf.Point(50, 60),
        "修正「桃園市犬貓收容及處理收費標準」第三條附表，內容如下。",
        font=FONT,
        fontsize=10,
    )
    tw.write_text(p1)
    p2 = doc.new_page(width=W, height=H)
    _draw_table(
        p2,
        [
            ["項目", "單位", "收費標準"],
            ["犬貓收容費", "每日", "200元"],
            ["犬貓安樂死費", "每隻", "500元"],
        ],
        [50.0, 230.0, 380.0, 560.0],
        60,
        30,
    )
    out = tmp_path / "t.pdf"
    doc.save(out)
    return out


def test_rows_lines_writes_each_data_row_as_header_value_pairs(tmp_path):
    pdf = _pdf(tmp_path)
    lines = rows_lines(pdf, 1)
    assert lines == [
        "項目：犬貓收容費｜單位：每日｜收費標準：200元",
        "項目：犬貓安樂死費｜單位：每隻｜收費標準：500元",
    ]
    assert rows_lines(pdf, 0) == []  # no table on page 1


def test_rows_text_replaces_only_table_pages(tmp_path):
    pdf = _pdf(tmp_path)
    text = rows_text(pdf, pages=["第一頁原文", "第二頁原文"])
    assert text.startswith("第一頁原文")
    assert "收費標準：500元" in text and "第二頁原文" not in text


def test_routes():
    assert ROUTES == ("base", "ocr", "rows")
