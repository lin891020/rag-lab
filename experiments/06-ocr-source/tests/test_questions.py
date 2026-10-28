"""03's 82 questions, copied verbatim; every gold table must exist in every OCR route's corpus."""

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


def test_copied_from_03_verbatim(questions):
    assert (HERE / "questions.jsonl").read_bytes() == (THREE / "questions.jsonl").read_bytes()
    assert len(questions) == 82


def test_every_gold_table_exists_in_the_built_routes(questions):
    routes = [p.name for p in (HERE / "corpus").iterdir() if p.is_dir() and p.name != "glm-raw"]
    assert "clean" in routes and "vision" in routes
    for q in questions:
        for g in q["gold"]:
            for route in routes:
                assert (HERE / "corpus" / route / f"{g}.txt").exists(), (q["id"], g, route)
