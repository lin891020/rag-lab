"""38 questions: 13 control rows on first pages, 25 on continuation pages. Every question's
page must exist in the corpus, and the gold value must appear on that page's naive reading
(so a wrong answer under `page` is the pipeline's fault, not a missing number)."""

import json
import re
from pathlib import Path

import pytest

HERE = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def questions():
    return [
        json.loads(ln)
        for ln in (HERE / "questions.jsonl").read_text("utf-8").splitlines()
        if ln.strip()
    ]


def test_counts(questions):
    assert len(questions) == 38
    assert sum(q["cat"] == "cont" for q in questions) == 25
    assert len({q["id"] for q in questions}) == 38


def test_page_file_exists_and_holds_the_gold_value(questions):
    for q in questions:
        p = HERE / "corpus/pages" / f"{q['gold'][0]}-p{q['page']}.txt"
        assert p.exists(), q["id"]
        v = q["expect"]["value"]
        s = f"{v:g}"
        assert re.search(rf"(?<![\d.]){re.escape(s)}(?![\d])", p.read_text("utf-8")), (q["id"], s)


def test_control_questions_sit_on_the_first_page(questions):
    assert all((q["page"] == 0) == (q["cat"] == "ctrl") for q in questions)


def test_six_control_questions_are_03s_verbatim(questions):
    three = {
        json.loads(ln)["id"]: json.loads(ln)
        for ln in (HERE.parent / "03-table-format/questions.jsonl").read_text("utf-8").splitlines()
        if ln.strip()
    }
    copied = [q for q in questions if q["src"].startswith("03:")]
    assert len(copied) == 6
    for q in copied:
        s = three[q["src"][3:]]
        assert (q["question"], q["expect"]["value"], q["expect"]["unit"]) == (
            s["question"],
            s["expect"]["value"],
            s["expect"]["unit"],
        )
