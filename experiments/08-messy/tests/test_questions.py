"""Table questions = 03's 82 verbatim; text questions = 01's 40 article questions verbatim.
Every gold table must exist in every built table route."""

import json
from pathlib import Path

import pytest

HERE = Path(__file__).parents[1]
ONE = HERE.parent / "01-chunking"
THREE = HERE.parent / "03-table-format"


def _rows(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in p.read_text("utf-8").splitlines() if ln.strip()]


@pytest.fixture(scope="module")
def table_questions():
    return _rows(HERE / "questions-table.jsonl")


def test_table_questions_copied_from_03_verbatim(table_questions):
    assert (HERE / "questions-table.jsonl").read_bytes() == (THREE / "questions.jsonl").read_bytes()
    assert len(table_questions) == 82


def test_text_questions_are_01s_article_questions_plus_reserve():
    qs = _rows(HERE / "questions-text.jsonl")
    ref = [
        q
        for name in ("questions.jsonl", "questions-reserve.jsonl")
        for q in _rows(ONE / name)
        if q["type"] == "article"
    ]
    assert qs == ref and len(qs) == 40


def test_every_gold_table_exists_in_the_built_routes(table_questions):
    tables = HERE / "corpus/tables"
    if not tables.exists():
        pytest.skip("corpus not built")
    routes = [p.name for p in tables.iterdir() if p.is_dir() and not p.name.endswith("-raw")]
    assert "clean" in routes
    for q in table_questions:
        for g in q["gold"]:
            for route in routes:
                assert (tables / route / f"{g}.txt").exists(), (q["id"], g, route)
