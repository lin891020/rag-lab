"""Post-hoc tables count what they say: abstentions per cell, and the rerun table compares the
re-asked rows with 16's copy (same raw text, same answer, flips in `right`)."""

import json

from posthoc import abstain, rerun

Q = {
    "a": {"id": "a", "expect": {"kind": "number", "value": 16.0, "unit": "A", "ladder": None}},
    "b": {"id": "b", "expect": {"kind": "number", "value": 20.0, "unit": "A", "ladder": None}},
}


def _ans(v: float) -> str:
    return json.dumps({"found": True, "quote": "", "expression": str(v), "value": v, "unit": "A"})


def _row(qid: str, v: float, *, reused: bool) -> dict:
    r = {"model": "gpt-oss:20b", "condition": "struct@1500", "qid": qid, "raw_content": _ans(v)}
    return {**r, "reused_from": "16"} if reused else r


def test_rerun_counts_same_text_same_answer_and_flips():
    raw = [
        _row("a", 16, reused=True),
        _row("b", 20, reused=True),
        _row("a", 16, reused=False),  # same
        _row("b", 23, reused=False),  # right -> wrong
    ]
    line = rerun(raw, Q)[-1]
    assert "| 1/2 | 1/2 |" in line and line.endswith("| 1 |")


def test_abstain_counts_unanswered_should_abstain_rows():
    rows = [
        {"model": "qwen3:8b", "condition": "cards", "kind": "abstain", "answered": False},
        {"model": "qwen3:8b", "condition": "cards", "kind": "abstain", "answered": True},
        {"model": "qwen3:8b", "condition": "cards", "kind": "number", "answered": False},
    ]
    assert "1/2" in abstain(rows)[2]
