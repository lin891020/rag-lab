"""The lookup tool reads cells of the clean tables and nothing else: exact and loose row / column
names, range rows and range columns by a number (the first row in table order that holds it),
blanks and dashes passed through, misses reported instead of guessed; and fed each question's
gold (table, row) it reaches the gold cell for every C1/C4/C5 question."""

import json
from pathlib import Path

import lookup
from lookup import TABLES, _fold, catalog, render, resolve
from rules import row_key

HERE = Path(__file__).parents[1]
Q = [json.loads(ln) for ln in (HERE / "questions.jsonl").read_text("utf-8").splitlines() if ln]


def test_exact_and_loose_row_and_column():
    a = resolve("表二五～三", "單線 2.0", "同一導線管或電纜內之載流導線數／7-9／安培容量（A）")
    b = resolve("表二五～三", "2.0mm 單線", "7-9")
    assert a["status"] == b["status"] == "ok"
    assert a["value"] == b["value"] == "16"
    assert a["label"] == "表二五～三|單線 2.0"


def test_conductor_count_as_a_number_picks_the_range_column():
    assert resolve("表二五～三", "單線 2.0", "8 條")["value"] == "16"
    assert (
        resolve("表二五～三", "單線 2.0", "3")["value"]
        == resolve("表二五～三", "單線 2.0", "3以下")["value"]
    )


def test_range_rows_take_the_first_row_that_holds_the_number():
    cases = {
        700: "800以下",
        800: "800以下",
        801: "1,000以下",
        1800: "2,000以下",
        20: "20以下",
        6300: "6,300以下",
    }
    for x, key in cases.items():
        r = resolve("表九三～二", f"{x} A", "絞線")
        assert r["label"] == f"表九三～二|{key}", x
    assert resolve("表九三～二", "7000", "絞線")["status"] == "no_row"


def test_range_rows_with_a_prefix():
    assert resolve("表五二", "住宅 70000", "需量因數")["label"] == "表五二|住宅 3,001至120,000"
    assert resolve("表五二", "住宅 3,000", "需量因數")["label"] == "表五二|住宅 3,000以下"
    assert resolve("表五二", "住宅 3001", "需量因數")["label"] == "表五二|住宅 3,001至120,000"
    assert resolve("表五二", "住宅 超過120,000", "需量因數")["value"] == "25"
    assert resolve("表五二", "醫院 50,000以下", "需量因數")["label"] == "表五二|醫院註 50,000以下"
    assert (
        resolve("表五二", "大賣場(倉儲) 100000", "需量因數")["label"]
        == "表五二|大賣場(倉儲) 超過12,500"
    )


def test_over_a_to_b_rows():
    assert resolve("表九三～一", "100", "銅導線線徑")["label"] == "表九三～一|超過 80 - 200"
    assert resolve("表九三～一", "80", "銅導線線徑")["label"] == "表九三～一|60 - 80"


def test_table_names_are_normalised():
    assert resolve("表六〇", "其他負載之剩餘部分", "需量因數")["value"] == "40"
    assert resolve("表六0", "其他負載之剩餘部分", "需量因數")["status"] == "ok"
    assert resolve("表二五-三 金屬導線管配線之導線安培容量", "單線 2.0", "4")["status"] == "ok"
    assert resolve("表九九", "1", "x")["status"] == "no_table"


def test_blank_dash_and_misses():
    assert resolve("表二五～二", "絞線 325", "5-6")["status"] == "blank"
    dash = resolve("表九三～二", "800以下", "單線")
    assert dash["status"] == "ok" and dash["value"] == "-"
    assert resolve("表二五～三", "單線 9.9", "4")["status"] == "no_row"
    assert resolve("表二五～三", "單線 2.0", "安培容量")["status"] == "no_column"  # ambiguous


def test_gold_rows_reach_the_gold_cell():
    """Every answerable C1/C4/C5 question: the gold rows exist, and the gold cell value is in one
    of the gold rows (the column is the model's job; this checks the tool can reach it)."""
    n = 0
    for q in Q:
        if q["chain"] not in ("C1", "C4", "C5") or q["expect"]["kind"] == "abstain":
            continue
        for tid, key in q["gold"]["rows"]:
            r = resolve(tid, key, lookup.TABLE_NAMES[tid][-1])
            assert r["status"] in ("ok", "blank") and r["label"] == f"{tid}|{key}", q["id"]
        n += 1
    assert n == 78


def test_render_has_titles_cells_notes_and_misses():
    rs = [resolve("表五二", "住宅 3,000以下", "需量因數"), resolve("表九九", "1", "x")]
    text = render(rs)
    assert "表五二 照明負載之幹線需量因數" in text.splitlines()[0]
    assert "表五二｜住宅 3,000以下｜需量因數（%） = 100" in text
    assert "表五二 註：" in text
    assert "查無此表：表九九" in text


def test_catalog_lists_every_table_without_values():
    cat = catalog()
    assert cat.count("\n表") + 1 == 22
    assert "單線 2.0" in cat and "800以下" in cat
    assert "239" not in cat  # 表二五～三 絞線 125, a value


def test_the_tool_does_not_use_the_rule_functions():
    for name in (
        "ampacity",
        "corrected_ampacity",
        "ground_conductor",
        "lighting_demand",
        "dwelling_other_demand",
        "receptacle_demand",
        "motor_branch_conductor",
    ):
        assert not hasattr(lookup, name)


def test_above_words_are_not_read_as_the_band_below():
    """'超過 10' written loosely must not land on 10以下 (review, 2026-11-01): the range path only
    takes a bare number (or 以下); anything else that is not the key is a miss, said as such."""
    for table, row in (
        ("表五三", "超過10 kVA之部分"),
        ("表五三", "超過10kVA部分"),
        ("表五三", "10 kVA以上"),
        ("表五二", "住宅 超過3,000"),
        ("表五二", "住宅 3000以上"),
        ("表九三～二", "超過600"),
    ):
        assert (
            resolve(table, row, "需量因數" if table != "表九三～二" else "絞線")["status"]
            == "no_row"
        ), row
    assert resolve("表五三", "超過10", "需量因數")["value"] == "50"  # the key itself still works
    assert resolve("表九三～二", "600以下", "絞線")["label"] == "表九三～二|600以下"


def test_stray_brackets_do_not_hide_a_name():
    """gpt-oss wrote 「需量因數（%}」 in the first run (15 questions lost to it); the fix cells read
    names with brackets ignored, the first run's strict matching is kept for its own rows."""
    assert resolve("表五二", "住宅 3,000以下", "需量因數（%}")["status"] == "ok"
    assert resolve("表五二", "住宅 3,000以下", "需量因數（%}", strict=True)["status"] == "no_column"
    assert (
        resolve("表二五～二", "絞線 8", "同一導線管或電纜內之載流導線數／7-9／安培容量（A}")[
            "value"
        ]
        == "25"
    )
    assert (
        resolve("表五二", "大賣場倉儲 12,500以下", "需量因數")["label"]
        == "表五二|大賣場(倉儲) 12,500以下"
    )


def test_names_stay_distinct_without_brackets():
    for tid, t in TABLES.items():
        keys = [_fold(row_key(tid, r)) for r in t["rows"]]
        raw = [_fold(row_key(tid, r), strict=True) for r in t["rows"]]
        assert len(set(keys)) == len(set(raw)), tid
        names = [_fold(n) for n in t["names"]]
        assert len(set(names)) == len({_fold(n, strict=True) for n in t["names"]}), tid
