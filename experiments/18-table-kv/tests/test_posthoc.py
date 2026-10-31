"""Post-hoc tables count what they say: the rerun table compares 18's re-asked rows with 17's
(same raw text, same answer, flips in `right`), and the token table skips copied rows."""

import json

from posthoc import rerun, tokens

Q = {
    "a": {"id": "a", "expect": {"kind": "number", "value": 16.0, "unit": "A", "ladder": None}},
    "b": {"id": "b", "expect": {"kind": "number", "value": 20.0, "unit": "A", "ladder": None}},
}


def _ans(v: float) -> str:
    return json.dumps({"found": True, "quote": "", "expression": str(v), "value": v, "unit": "A"})


def _row(qid: str, v: float, **extra) -> dict:
    return {
        "model": "gpt-oss:20b",
        "condition": "cards",
        "qid": qid,
        "raw_content": _ans(v),
        **extra,
    }


def test_rerun_counts_same_text_same_answer_and_flips():
    old = [_row("a", 16), _row("b", 20)]
    new = [_row("a", 16), _row("b", 23), _row("b", 20, reused_from="17")]
    line = rerun(old, new, Q, "gpt-oss:20b", "cards")[-1]
    assert "| 1/2 | 1/2 |" in line and line.endswith("| 1 |")


def test_tokens_skip_copied_rows():
    raw = [
        _row("a", 16, prompt_len=100, prompt_eval_count=90, done_reason="stop"),
        _row("b", 20, prompt_len=999, prompt_eval_count=999, done_reason="stop", reused_from="17"),
    ]
    line = tokens(raw)[-1]
    assert "| 100 | 90 |" in line and "stop 1" in line
