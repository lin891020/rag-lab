"""Grid extraction with exact spans (from pdfplumber cell geometry) and the three renderings.
Fixtures are real attachment PDFs: 表八六～三 (two header rows, colspans, a split corner),
表二五～七 (empty bottom-right cells), 表五六 (two pages, rowspans in data), 表二五～二 (rows
not ruled apart: one cell holds 18 lines)."""

from pathlib import Path

import pytest

from tables import load_table, to_html, to_kv, to_markdown

FIX = Path(__file__).parents[3] / "tests/fixtures"


@pytest.fixture(scope="module")
def t86():
    return load_table(FIX / "table86-3.pdf")


@pytest.fixture(scope="module")
def t25_7():
    return load_table(FIX / "table25-7.pdf")


@pytest.fixture(scope="module")
def t56():
    return load_table(FIX / "table56.pdf")


@pytest.fixture(scope="module")
def t25_2():
    return load_table(FIX / "table25-2.pdf")


def test_title_notes_and_header_depth(t86):
    assert t86.title == "表八六～三 低壓用戶斷路器之額定極限短路啟斷容量"
    assert t86.notes.startswith("註：1.本表啟斷容量") and "50 %以上" in t86.notes
    assert t86.header_rows == 2
    assert len(t86.data_rows()) == 3


def test_split_corner_column_is_dropped_and_names_join_header_levels(t86):
    names = t86.column_names()
    assert len(names) == 10
    assert names[0] == "最低額定短路容量裝設位"
    assert names[1] == "單相110 V、220 V用戶／75以下"
    assert names[3] == "單相110 V、220 V用戶／超過100"
    assert names[9] == "三相380 V用戶／超過200"


def test_markdown_rows(t86):
    md = to_markdown(t86)
    lines = md.splitlines()
    assert lines[0] == "表八六～三 低壓用戶斷路器之額定極限短路啟斷容量"
    assert lines[1].startswith("| 最低額定短路容量裝設位 | 單相110 V、220 V用戶／75以下 |")
    assert "| 集中(單獨)表箱 | 20 | 20 | 25 | 20 | 20 | 25 | 25 | 25 | 30 |" in lines
    assert lines[-1].startswith("註：")


def test_kv_rows(t86):
    kv = to_kv(t86)
    assert "最低額定短路容量裝設位=集中(單獨)表箱；單相110 V、220 V用戶／75以下=20；" in kv
    assert "三相380 V用戶／超過200=30" in kv


def test_html_has_real_spans(t86):
    html = to_html(t86)
    assert '<th rowspan="2">最低額定短路容量裝設位</th>' in html
    assert '<th colspan="3">單相110 V、220 V用戶</th>' in html
    assert "<th>75以下</th>" in html
    assert "<td>受電箱</td><td>35</td>" in html
    assert html.count("<tr>") == 5


def test_empty_cells_stay_empty(t25_7):
    assert t25_7.header_rows == 2
    names = t25_7.column_names()
    assert names == [
        "周圍溫度（℃）",
        "導線絕緣物最高容許溫度（℃）／60",
        "導線絕緣物最高容許溫度（℃）／75",
        "導線絕緣物最高容許溫度（℃）／90",
    ]
    rows = {r[0]: r for r in t25_7.data_rows()}
    assert rows["41 - 45"] == ["41 - 45", "0.77", "0.87", "0.90"]
    assert rows["76 - 80"] == ["76 - 80", "", "", "0.43"]
    assert "周圍溫度（℃）=76 - 80；導線絕緣物最高容許溫度（℃）／90=0.43" in to_kv(t25_7)


def test_two_pages_and_rowspans_in_data(t56):
    rows = t56.data_rows()
    assert len(rows) == 30
    assert rows[11][:2] == ["12", "27"]
    by = {r[0]: r for r in rows}
    assert by["31 - 40"] == ["31 - 40", "15 ＋電爐數 × 1", "30", "22"]
    assert by["61以上"] == ["61以上", "25 ＋電爐數 × 0.75", "30", "16"]
    assert t56.notes.startswith("註：1.相同額定之電爐")


def test_unruled_rows_are_exploded(t25_2):
    assert t25_2.header_rows == 3
    rows = t25_2.data_rows()
    assert len(rows) == 21
    assert rows[0] == ["單線", "", "1.6", "13", "12", "11", "9"]
    assert rows[7] == ["絞線", "22", "7/2.0", "65", "59", "52", "46"]
    assert rows[-1] == ["絞線", "500", "61/3.2", "409", "", "", ""]
    names = t25_2.column_names()
    assert names[1] == "導線線徑／標稱截面積（mm2)"
    assert names[3] == "同一導線管或電纜內之載流導線數／3以下／安培容量（A）"


def test_text_rows_with_many_lines_are_not_exploded():
    t = load_table(FIX / "table92.pdf")
    rows = t.data_rows()
    assert [r[0] for r in rows] == ["特種接地", "第一種接地", "第二種接地", "第三種接地"]
    assert rows[3][2].startswith("1.對地電壓150 V以下：100以下2.對地電壓151 V至300 V：50以下")
