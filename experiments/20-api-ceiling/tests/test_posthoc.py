"""posthoc's step-1 class: which stage of the lookup a reply lost at (table, row, column), and the
gold cells each chain needs (C4's 'first 10 kW' band is 100 % without an input of its own)."""

import json
from pathlib import Path

from lookup import resolve
from posthoc import gold_cells, step1_class

HERE = Path(__file__).parents[1]
Q = {
    q["id"]: q for q in map(json.loads, (HERE / "questions.jsonl").read_text("utf-8").splitlines())
}


def _row(*lookups):
    return {"lookups": [resolve(*x) for x in lookups]}


def test_gold_cells_per_chain():
    assert gold_cells(Q["c1-10"]) == [20.0]
    assert gold_cells(Q["c5-10"]) == [125.0]
    assert sorted(gold_cells(Q["c4-20"])) == [50.0, 100.0]
    assert sorted(gold_cells(Q["c4-10"])) == [40.0, 100.0]


def test_step1_class_stages():
    q = Q["c1-10"]  # 表二五～五 絞線 8, 9 conductors -> 20
    assert step1_class({"lookups": []}, q) == "no_lookups"
    assert step1_class(_row(("表二五～二", "絞線 8", "7-9")), q) == "wrong_table"
    assert step1_class(_row(("表二五～五", "絞線 14", "7-9")), q) == "wrong_row"
    assert step1_class(_row(("表二五～五", "絞線 8", "3以下")), q) == "wrong_column"
    assert step1_class(_row(("表二五～五", "絞線 8", "9")), q) == "cells_ok"
    c4 = Q["c4-20"]  # 大賣場 two bands, 100 % and 50 %
    one = _row(("表五二", "大賣場(倉儲) 100000", "需量因數"))
    assert step1_class(one, c4) == "wrong_row"  # the first band was not looked up
    both = _row(
        ("表五二", "大賣場(倉儲) 12,500以下", "需量因數"),
        ("表五二", "大賣場(倉儲) 超過12,500", "需量因數"),
    )
    assert step1_class(both, c4) == "cells_ok"


def test_a_band_with_nothing_in_it_is_not_required():
    q = Q["c4-16"]  # 6 kVA of receptacles: only the 10以下 band is used
    assert step1_class(_row(("表五三", "10以下", "需量因數")), q) == "cells_ok"


def test_right_row_with_an_unknown_column_is_a_column_miss():
    q = Q["c1-10"]
    assert step1_class(_row(("表二五～五", "絞線 8", "沒有這欄")), q) == "wrong_column"
