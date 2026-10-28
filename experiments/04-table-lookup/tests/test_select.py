"""Row selection must find the right row from the question text alone (no gold)."""

from pathlib import Path

import pytest

from select_rows import question_numbers, select_rows, subtable_markdown
from tables import load_table

FIX = Path(__file__).parents[3] / "tests/fixtures"


@pytest.fixture(scope="module")
def t93():
    return load_table(FIX / "table93-1.pdf")


@pytest.fixture(scope="module")
def t25_2():
    return load_table(FIX / "table25-2.pdf")


def test_question_numbers_handles_commas_and_decimals():
    assert question_numbers("住宅照明負載 50,000 瓦，2.0 mm 的導線，3 條") == [50000.0, 2.0, 3.0]


def test_number_inside_a_range_row_wins(t93):
    rows = t93.data_rows()
    picked = select_rows(t93, "進屋線 50 平方，內線系統接地的接地電極導線要幾平方毫米？", k=3)
    assert rows[picked[0]][0] == "38 - 50"
    picked = select_rows(t93, "進屋線 100 平方，內線系統接地的接地電極導線要幾平方毫米？", k=3)
    assert rows[picked[0]][0] == "超過 80 - 200"


def test_bigram_match_finds_the_row_in_a_wide_table(t25_2):
    rows = t25_2.data_rows()
    picked = select_rows(
        t25_2, "60 度 PVC 導線走金屬管，3 條載流導線，22 平方的安培容量是多少？", k=3
    )
    assert any(rows[i][1] == "22" for i in picked)
    assert len(picked) == 3 and picked == sorted(picked)


def test_k_caps_and_small_tables_return_everything(t93):
    assert len(select_rows(t93, "幾平方？", k=3)) == 3
    assert select_rows(t93, "幾平方？", k=10) == list(range(6))


def test_subtable_markdown_keeps_header_and_only_picked_rows(t93):
    md = subtable_markdown(t93, [3])
    lines = md.splitlines()
    assert lines[1].startswith("| 進屋線或變壓器")
    assert "| 超過 80 - 200 | 30 |" in md and "| 38 - 50 |" not in md
    assert lines[-1].startswith("註：")
