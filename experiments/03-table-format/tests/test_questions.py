"""questions.jsonl is hand-made data: complete rows, known units, gold tables present in every
rendering, and the 02 rows carried over unchanged."""

import json
from pathlib import Path

import pytest

from score import UNIT_ALIASES

HERE = Path(__file__).parents[1]
ONE = HERE.parent / "01-chunking"
TWO = HERE.parent / "02-json-output"


@pytest.fixture(scope="module")
def questions():
    return [
        json.loads(ln)
        for ln in (HERE / "questions.jsonl").read_text("utf-8").splitlines()
        if ln.strip()
    ]


def test_ids_unique_and_enough(questions):
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids)) and len(ids) >= 60


def test_expect_is_numeric_with_known_unit(questions):
    for q in questions:
        e = q["expect"]
        assert isinstance(e["value"], (int, float)) and not isinstance(e["value"], bool), q["id"]
        assert e["unit"] in UNIT_ALIASES, (q["id"], e["unit"])
        for a in e["alt"]:
            assert a["unit"] in UNIT_ALIASES, q["id"]


def test_every_gold_table_exists_in_all_four_renderings(questions):
    for q in questions:
        assert q["type"] == "table"
        for g in q["gold"]:
            assert (ONE / "corpus/tables" / f"{g}.txt").exists(), (q["id"], g)
            for fmt in ("markdown", "html", "kv"):
                assert (HERE / "corpus" / fmt / f"{g}.txt").exists(), (q["id"], fmt, g)


def test_rows_carried_from_02_are_unchanged(questions):
    two = {
        json.loads(ln)["id"]: json.loads(ln)
        for ln in (TWO / "questions.jsonl").read_text("utf-8").splitlines()
        if ln.strip()
    }
    carried = [q for q in questions if q["src"].startswith("02:")]
    assert len(carried) == 22
    for q in carried:
        src = two[q["src"].split(":")[1]]
        assert (q["question"], q["gold"], q["expect"]) == (
            src["question"],
            src["gold"],
            src["expect"],
        ), q["id"]
