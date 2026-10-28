"""04 adds two things to 03's tables.py: parsing range cells and expanding them into bound columns.
Every spelling below occurs in the 17 tables' key columns (listed from the corpus, 2026-10-28)."""

from pathlib import Path

import pytest

from tables import expand_ranges, load_table, parse_range, to_markdown

FIX = Path(__file__).parents[3] / "tests/fixtures"


@pytest.mark.parametrize(
    "text, want",
    [
        ("38 - 50", (38, 50)),
        ("26 - 30", (26, 30)),
        ("1 - 4", (1, 4)),
        ("超過 80 - 200", (80, 200)),
        ("超過 325 - 500", (325, 500)),
        ("30以下", (None, 30)),
        ("10以下", (None, 10)),
        ("6,300以下", (None, 6300)),
        ("50,000以下", (None, 50000)),
        ("41以上", (41, None)),
        ("61以上", (61, None)),
        ("超過120,000", (120000, None)),
        ("3,001至120,000", (3001, 120000)),
        ("20,001至100,000", (20001, 100000)),
        ("5", (5, 5)),
        ("1,000", (1000, 1000)),
        ("2.0", (2.0, 2.0)),
    ],
)
def test_parse_range_known_spellings(text, want):
    assert parse_range(text) == want


@pytest.mark.parametrize(
    "text", ["", "-", "15或20", "住宅", "總伏安", "l00", "7/0.8", "單線", "最大15"]
)
def test_parse_range_refuses_non_ranges(text):
    assert parse_range(text) is None


def test_expand_adds_bound_columns_after_a_range_column():
    t = expand_ranges(load_table(FIX / "table93-1.pdf"))
    names = t.column_names()
    assert names[:3] == [
        "進屋線或變壓器二次側電源導線之最大截面積註1(mm2)",
        "進屋線或變壓器二次側電源導線之最大截面積註1(mm2)／下限",
        "進屋線或變壓器二次側電源導線之最大截面積註1(mm2)／上限",
    ]
    rows = {r[0]: r for r in t.data_rows()}
    assert rows["30以下"][:3] == ["30以下", "", "30"]
    assert rows["超過 80 - 200"][:3] == ["超過 80 - 200", "80", "200"]
    assert rows["超過 325 - 500"][3] == "60"  # the value column moved right, untouched
    md = to_markdown(t)
    assert "| 38 - 50 | 38 | 50 | 14 |" in md


def test_expand_leaves_single_number_columns_alone():
    t = expand_ranges(load_table(FIX / "table25-2.pdf"))
    assert "下限" not in "".join(t.column_names())  # 3.5 … 500 are points, not ranges


def test_expand_handles_open_ended_rows_on_the_second_page():
    t = expand_ranges(load_table(FIX / "table56.pdf"))
    rows = {r[0]: r for r in t.data_rows()}
    assert rows["61以上"][:3] == ["61以上", "61", ""]
    assert rows["12"][:3] == ["12", "12", "12"]
