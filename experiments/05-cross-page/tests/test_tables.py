"""Cross-page loading: per-page grids, the continuation rule, and the merge."""

from pathlib import Path

import pytest
from tables import (
    is_continuation,
    load_page,
    load_pages,
    load_table,
    merge_pages,
    to_markdown,
)

FIX = Path(__file__).parents[3] / "tests/fixtures"


@pytest.mark.parametrize(
    ("name", "n"),
    [("table56.pdf", 2), ("table587-1.pdf", 2), ("table587-3.pdf", 3), ("table8.pdf", 1)],
)
def test_load_pages_one_grid_per_page_with_a_table(name, n):
    grids = load_pages(FIX / name)
    assert len(grids) == n
    assert grids[0].has_title and all(not g.has_title for g in grids[1:])


def test_continuation_rule_accepts_the_three_real_cases():
    for name in ("table56.pdf", "table587-1.pdf", "table587-3.pdf"):
        grids = load_pages(FIX / name)
        for prev, cur in zip(grids, grids[1:], strict=False):
            assert is_continuation(prev, cur), name


def test_continuation_rule_rejects_another_tables_first_page():
    last = load_pages(FIX / "table56.pdf")[-1]
    first = load_pages(FIX / "table8.pdf")[0]
    assert not is_continuation(last, first)


def test_merge_carries_a_cut_rowspan_label_over_the_page_break():
    t = load_table(FIX / "table587-1.pdf", header_rows=1)
    rows = t.data_rows()
    hit = [r for r in rows if r[1] == "排放口" and "1.5 m至3 m" in r[4]]
    assert [r[0] for r in hit] == ["地上燃料槽", "地下燃料槽"]


def test_merge_joins_a_cell_split_by_the_page_break():
    t = load_table(FIX / "table587-3.pdf", header_rows=1)
    texts = {c for r in t.data_rows() for c in r}
    assert "內含閥門、配件或管線，且不位於第一種場所或第二種場所、1 區或2區" in texts
    assert not any(c == "場所、1 區或2區" for c in texts)


def test_merge_aligns_a_page_with_more_columns_on_the_union_of_edges():
    t = load_table(FIX / "table587-3.pdf", header_rows=1)
    names = t.column_names()
    zone = names.index("區")
    rows = t.data_rows()
    tank = [r for r in rows if r[0] == "罐槽車註2" and "圓蓋邊緣展開900 mm 範圍內" in r[-1]]
    assert len(tank) == 1 and tank[0][zone] == "1"
    first = [
        r
        for r in rows
        if r[0] == "設備裝設於室外，於正常運作條件下可能存在易燃性揮發氣與氣體混合物"
    ]
    assert first and first[0][zone] == "0"


def test_merge_keeps_the_notes_from_the_last_page():
    t = load_table(FIX / "table587-3.pdf", header_rows=1)
    assert t.notes.startswith("註：1.易燃性液體")


def test_naive_page_reading_has_no_title_and_promotes_the_first_row():
    t = load_page(FIX / "table56.pdf", 1)
    assert t.title == "" and t.column_names()[0] == "26 - 30"
    assert t.data_rows()[0][0] == "31 - 40"
    md = to_markdown(t)
    assert md.startswith("| 26 - 30 |") and "41 - 50" in md


def test_merge_pages_without_continuation_keeps_only_the_first_grid():
    a = load_pages(FIX / "table56.pdf")
    b = load_pages(FIX / "table8.pdf")
    cells, pages = merge_pages([a[0], b[0]])
    assert pages == [0]
