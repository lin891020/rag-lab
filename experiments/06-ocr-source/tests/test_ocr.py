"""Segment clustering, markdown rendering, HTML expansion and the tesseract CJK glue."""

from pathlib import Path

import pytest
from ocr import (
    Seg,
    cluster_rows,
    html_to_markdown,
    norm,
    rows_to_markdown,
    segments_to_markdown,
    vision_segments,
)

PNG = Path(__file__).parents[3] / "data/06-ocr/png"


def test_cluster_rows_groups_by_vertical_centre_and_orders_by_x():
    segs = [
        Seg("b", 100, 10, 120, 20),
        Seg("a", 10, 11, 30, 21),
        Seg("c", 10, 40, 30, 50),
    ]
    rows = cluster_rows(segs)
    assert [[s.text for s in r] for r in rows] == [["a", "b"], ["c"]]


def test_rows_to_markdown_keeps_title_first_and_notes_last():
    rows = [
        [Seg("表九三～一", 0, 0, 10, 5), Seg("接地", 20, 0, 30, 5)],
        [Seg("截面積", 0, 10, 10, 15), Seg("線徑", 20, 10, 30, 15)],
        [Seg("38 - 50", 0, 20, 10, 25), Seg("14", 20, 20, 30, 25)],
        [Seg("註：1. 進屋線", 0, 30, 10, 35)],
        [Seg("2. 超過500", 0, 40, 10, 45)],
    ]
    md = rows_to_markdown(rows)
    assert md.splitlines()[0] == "表九三～一接地"
    assert md.splitlines()[1:4] == ["| 截面積 | 線徑 |", "|---|---|", "| 38 - 50 | 14 |"]
    assert md.splitlines()[-1] == "註：1. 進屋線2. 超過500"


def test_html_to_markdown_expands_spans():
    html = (
        "<table><tr><th rowspan='2'>a</th><th colspan='2'>b</th></tr>"
        "<tr><th>c</th><th>d</th></tr><tr><td>1</td><td>2</td><td>3</td></tr></table>"
    )
    md = html_to_markdown(html)
    assert md.splitlines() == ["| a | b | b |", "|---|---|---|", "| a | c | d |", "| 1 | 2 | 3 |"]


def test_norm_folds_width_and_spaces():
    assert norm("３８ - ５０") == norm("38-50") == "38-50"
    assert norm("（mm）") == "(mm)"


@pytest.mark.skipif(not PNG.exists(), reason="rendered pages live in data/, not in git")
def test_vision_reads_the_range_table():
    md = segments_to_markdown(vision_segments(PNG / "表九三～一-p0.png"))
    assert md.startswith("表九三～一")
    assert "| 38 - 50 | 14 |" in md and "| 超過 325 - 500 | 60 |" in md
    assert md.splitlines()[-1].startswith("註")
