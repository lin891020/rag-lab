"""posthoc24's counts: a reply's value, the same answer as another condition, flips between two
conditions, and which questions flattening changed the prompt of."""

import json

from posthoc24 import answer, changed_questions, flips, same_answers, same_replies


def _r(cond, qid, value, ok, model="m"):
    content = json.dumps({"quote": "q", "value": value, "unit": "A"})
    return {"model": model, "condition": cond, "qid": qid, "content": content, "value_ok": ok}


ROWS = [
    _r("clean", "a", 18, True),
    _r("glm2", "a", 18.0, True),
    _r("glm2-flat", "a", 18, True),
    _r("clean", "b", 5, True),
    _r("glm2", "b", 6, False),
    _r("glm2-flat", "b", 5, True),
    _r("clean", "c", 7, False),
    _r("glm2", "c", 9, True),
    _r("glm2-flat", "c", 7, False),
]


def test_answer_reads_the_value_or_none():
    assert answer('{"quote": "", "value": 18, "unit": "A"}') == 18.0
    assert answer('```json\n{"quote": "", "value": "1,000", "unit": "A"}\n```') == 1000.0
    assert answer("not json") is None
    assert answer('{"quote": "", "value": null, "unit": ""}') is None


def test_same_answers_compares_values_on_the_given_questions():
    assert same_answers(ROWS, ROWS, "m", "glm2", "clean", ["a", "b", "c"]) == 1
    assert same_answers(ROWS, ROWS, "m", "glm2-flat", "clean", ["a", "b", "c"]) == 3
    assert same_answers(ROWS, ROWS, "m", "glm2-flat", "clean", ["b"]) == 1


def test_same_replies_is_byte_equality():
    other = [dict(r, content=r["content"] + " ") if r["qid"] == "a" else r for r in ROWS]
    assert same_replies(ROWS, other, "m", "glm2", "glm2", ["a", "b"]) == 1


def test_flips_from_before_to_after():
    assert flips(ROWS, "m", "glm2", "glm2-flat") == {"won": ["b"], "lost": ["c"]}


def test_changed_questions_are_those_whose_gold_table_text_differs():
    qs = [{"id": "a", "gold": ["t1"]}, {"id": "b", "gold": ["t2"]}]
    corpus = {"glm2": {"t1": "x", "t2": "y"}, "glm2-flat": {"t1": "x", "t2": "z"}}
    assert changed_questions(qs, corpus, "glm2", "glm2-flat") == {"b"}


def test_two_unparseable_replies_are_not_the_same_answer():
    bad = [dict(r, content="oops") for r in ROWS if r["qid"] == "a"]
    assert same_answers(bad, bad, "m", "glm2", "clean", ["a"]) == 0
