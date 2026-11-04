"""posthoc25's counts: the same answer in both runs of a cell, and the cost of each cell."""

import json

from posthoc25 import cost, noise


def _r(model, cond, qid, value, ok, tin=1000, tout=100, err=""):
    content = json.dumps({"quote": "q", "value": value, "unit": "A"}) if not err else ""
    return {
        "model": model,
        "condition": cond,
        "qid": qid,
        "content": content,
        "value_ok": ok,
        "input_tokens": tin,
        "output_tokens": tout,
        "duration": 2.0,
        "error": err,
    }


ROWS = [
    _r("claude-opus-5-5", "image", "a", 1, True),
    _r("claude-opus-5-5", "image-2", "a", 1.0, True),
    _r("claude-opus-5-5", "image", "b", 2, False),
    _r("claude-opus-5-5", "image-2", "b", 3, True),
]


def test_noise_counts_same_answers_and_the_accuracy_change():
    n = noise(ROWS, "claude-opus-5-5", "image", ["a", "b"])
    assert n == {"same": 1, "n": 2, "ok1": 1, "ok2": 2, "raw_same": 0}  # 1 and 1.0 differ as text


def test_cost_sums_every_call_failed_ones_too():
    rows = ROWS + [_r("claude-opus-5-5", "image", "c", 0, False, tin=500, tout=0, err="stop")]
    c = cost(rows, "claude-opus-5-5", ["image", "image-2"])
    assert c["calls"] == 5 and c["usd"] == round((4500 * 4 + 400 * 20) / 1_000_000, 4)
    assert c["per_question"] == round(c["usd"] / 5, 5)  # 5 (condition, question) keys
