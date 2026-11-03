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


def test_shown_finds_the_gold_cells_in_a_whole_column():
    from lookup import render_segment
    from posthoc import read_slip, shown

    q = Q["c4-20"]  # 大賣場 two bands, 100 % and 50 %
    text, _ = render_segment([resolve("表五二", "大賣場(倉儲) 100000", "需量因數")])
    assert shown({"evidence_text": text}, q)
    one = "表五二｜大賣場(倉儲) 超過12,500｜需量因數(%) = 50"
    assert not shown({"evidence_text": one}, q)
    slip = {"evidence_text": text, "expression": "100000 × 0.25", "right": False}
    assert read_slip(slip, q)  # 25 is another row's value, not one of this question's cells
    fine = {"evidence_text": text, "expression": "12500 × 1 + 87500 × 0.5", "right": False}
    assert not read_slip(fine, q)


def test_a_percent_divisor_or_a_missing_band_is_not_a_slip():
    from lookup import render_segment
    from posthoc import read_slip

    q = Q["c4-28"]  # 旅館 30,000 W: 20,000 at 50 % + 10,000 at 40 %
    text, _ = render_segment(
        [
            resolve(
                "表五二",
                "飯店、旅館及汽車旅館，包括不提供房客烹飪用電器具之公寓式房屋註 20,000以下",
                "需量因數",
            )
        ]
    )
    one_band = {"evidence_text": text, "expression": "30000 × 50 ÷ 100", "right": False}
    assert not read_slip(one_band, q)  # one factor for the whole load: a rule slip, not a row slip


def test_another_column_of_the_gold_row_is_not_a_row_slip():
    from lookup import render_segment
    from posthoc import read_slip, shown

    q = Q["c1-12"]  # 表二五～三 單線 2.0, 8 conductors -> 16; its 4-conductor cell says 21
    asked = [
        ("表二五～三", "單線 2.0", "4"),
        ("表二五～三", "單線 2.0", "8"),
        ("表二五～四", "單線 2.0", "4"),
    ]
    text, _ = render_segment([resolve(*a) for a in asked])  # 19 qwen3 + the gold column
    assert shown({"evidence_text": text}, q) and "表二五～四｜單線 1.6｜" in text  # 21 there too
    assert not read_slip({"evidence_text": text, "expression": "21", "right": False}, q)
