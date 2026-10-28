"""questions.jsonl is hand-made data: every row must be complete and point at real 01 material."""

import json
from pathlib import Path

import pytest

from score import UNIT_ALIASES

HERE = Path(__file__).parents[1]
ONE = HERE.parent / "01-chunking"


@pytest.fixture(scope="module")
def questions():
    return [
        json.loads(ln)
        for ln in (HERE / "questions.jsonl").read_text("utf-8").splitlines()
        if ln.strip()
    ]


def test_ids_unique_and_enough(questions):
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))
    assert len(ids) >= 35


def test_every_row_has_numeric_expect_and_known_unit(questions):
    for q in questions:
        e = q["expect"]
        assert isinstance(e["value"], (int, float)) and not isinstance(e["value"], bool), q["id"]
        assert e["unit"] in UNIT_ALIASES, (q["id"], e["unit"])
        for a in e["alt"]:
            assert isinstance(a["value"], (int, float)) and a["unit"] in UNIT_ALIASES, q["id"]


def test_gold_exists_in_01_corpus(questions):
    import sys

    sys.path.insert(0, str(ONE))
    from chunkers import parse_articles  # noqa: E402

    from rag_lab.pdf import extract_text

    nums = {str(a.num) for a in parse_articles(extract_text(ONE / "corpus/law.pdf"))}
    for q in questions:
        for g in q["gold"]:
            if q["type"] == "article":
                assert g in nums, (q["id"], g)
            else:
                assert (ONE / "corpus/tables" / f"{g}.txt").exists(), (q["id"], g)


def test_src_points_at_a_01_question(questions):
    pool = {}
    for name, tag in (("questions.jsonl", "01"), ("questions-reserve.jsonl", "01r")):
        for ln in (ONE / name).read_text("utf-8").splitlines():
            if ln.strip():
                pool[f"{tag}:{json.loads(ln)['id']}"] = True
    for q in questions:
        assert q["src"] in pool, (q["id"], q["src"])
