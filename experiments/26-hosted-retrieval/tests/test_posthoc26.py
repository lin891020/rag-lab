"""posthoc26's counts: the same answer (found and value), right split by whether the gold rows
were all in the evidence, and cost per question answered."""

from posthoc26 import by_hit, cost, same_answer


def test_same_answer_compares_found_and_value():
    a = '{"found": true, "quote": "x", "expression": "", "value": 8, "unit": "mm²"}'
    b = '{"found": true, "quote": "y", "expression": "", "value": 8.0, "unit": "mm2"}'
    c = '{"found": false, "quote": "", "expression": "", "value": 0, "unit": ""}'
    assert same_answer(a, b) and not same_answer(a, c) and same_answer(c, c)
    assert not same_answer("", "")  # two stopped replies are no answer


def test_by_hit_splits_answerable_rows():
    rows = [
        {"model": "m", "condition": "cards", "kind": "number", "rows_all_hit": True, "right": True},
        {
            "model": "m",
            "condition": "cards",
            "kind": "number",
            "rows_all_hit": True,
            "right": False,
        },
        {
            "model": "m",
            "condition": "cards",
            "kind": "ladder",
            "rows_all_hit": False,
            "right": False,
        },
        {
            "model": "m",
            "condition": "cards",
            "kind": "abstain",
            "rows_all_hit": False,
            "right": True,
        },
    ]
    assert by_hit(rows, "m", "cards") == {"hit": (1, 2), "miss": (0, 1)}


def test_cost_divides_by_questions_answered():
    rows = [
        {
            "model": "claude-opus-5-5",
            "condition": "cards",
            "qid": "a",
            "input_tokens": 1000,
            "output_tokens": 100,
            "error": "",
            "duration": 1.0,
        },
        {
            "model": "claude-opus-5-5",
            "condition": "cards",
            "qid": "a",
            "input_tokens": 0,
            "output_tokens": 0,
            "error": "x",
            "duration": 0.0,
        },
        {
            "model": "claude-opus-5-5",
            "condition": "cards",
            "qid": "b",
            "input_tokens": 1000,
            "output_tokens": 100,
            "error": "",
            "duration": 1.0,
        },
    ]
    c = cost(rows, "claude-opus-5-5", ["cards"])
    assert c["calls"] == 3 and c["usd"] == round((2000 * 4 + 200 * 20) / 1e6, 4)
    assert c["per_question"] == round(c["usd"] / 2, 5)
