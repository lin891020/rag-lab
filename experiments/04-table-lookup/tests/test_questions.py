"""82 rows copied from 03 unchanged, plus a `row` key (the gold row's leading cells, in order) and a
category. The row key must match a real data row of the table in corpus/tables.json."""

import json
from pathlib import Path

import pytest

HERE = Path(__file__).parents[1]
THREE = HERE.parent / "03-table-format"


@pytest.fixture(scope="module")
def questions():
    return [
        json.loads(ln)
        for ln in (HERE / "questions.jsonl").read_text("utf-8").splitlines()
        if ln.strip()
    ]


@pytest.fixture(scope="module")
def tables():
    return json.loads((HERE / "corpus/tables.json").read_text("utf-8"))


def test_copied_from_03_verbatim(questions):
    three = {
        json.loads(ln)["id"]: json.loads(ln)
        for ln in (THREE / "questions.jsonl").read_text("utf-8").splitlines()
        if ln.strip()
    }
    assert len(questions) == 82 == len(three)
    for q in questions:
        s = three[q["id"]]
        assert (q["question"], q["gold"], q["expect"]) == (
            s["question"],
            s["gold"],
            s["expect"],
        ), q["id"]
        assert q["cat"] in {"range", "wide", "range+wide", "plain"}


def test_row_key_identifies_exactly_one_data_row(questions, tables):
    for q in questions:
        rows = tables[q["gold"][0]]["rows"]
        hits = [r for r in rows if r[: len(q["row"])] == q["row"]]
        assert len(hits) == 1, (q["id"], q["row"], len(hits))
