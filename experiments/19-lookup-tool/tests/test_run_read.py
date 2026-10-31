"""Copied baselines are the right rows, and one lookup question runs end to end with a fake chat
model (no ollama): step 1's cells are resolved, rendered as step 2's evidence, and the gold-row
hit is counted from what was looked up."""

import json

import run_read
from run_read import load_questions, parse_lookups, reused_rows, same_answer

Q = load_questions()


def test_questions_are_the_three_chains():
    assert len(Q) == 96 and {q["chain"] for q in Q} == {"C1", "C4", "C5"}


def test_reused_rows_cards_from_18_and_oracle_from_17():
    rows = reused_rows({q["id"] for q in Q}, run_read.MODELS)
    by = {(r["condition"], r["reused_from"]) for r in rows}
    assert by == {("cards", "18"), ("oracle", "17")}
    assert len(rows) == 2 * 2 * 96
    assert all(r["evidence_text"] for r in rows)


def test_parse_lookups_caps_and_fills():
    many = json.dumps({"lookups": [{"table": "表五二", "row": str(i)} for i in range(9)]})
    got, problem = parse_lookups(many)
    assert len(got) == 6 and problem == "over the limit" and got[0]["column"] == ""
    assert parse_lookups("not json") == ([], "unparseable")


def test_same_answer_ignores_wording():
    a = '{"found": true, "quote": "x", "expression": "16", "value": 16, "unit": "A"}'
    b = '{"found":true,"quote":"y","expression":"16","value":16,"unit":"A"}'
    assert same_answer(a, b) and not same_answer(a, b.replace("16,", "17,"))


def test_ask_lookup_end_to_end(monkeypatch):
    q = next(q for q in Q if q["id"] == "c1-10")  # PVC, 8 mm² 絞線, 60 ℃, 9 conductors -> 20
    replies = iter(
        [
            '{"lookups": [{"table": "表二五～五", "row": "絞線 8", "column": "9"}]}',
            '{"found": true, "quote": "", "expression": "20", "value": 20, "unit": "A"}',
        ]
    )
    prompts = []

    def fake(prompt, **kw):
        prompts.append(prompt)
        return {
            "content": next(replies),
            "thinking": "",
            "eval_count": 1,
            "prompt_eval_count": 10,
            "duration": 0.5,
            "done_reason": "stop",
        }

    monkeypatch.setattr(run_read.ollama, "chat", fake)
    row = run_read.ask_lookup("qwen3:8b", "lookup", q, run_read.catalog())
    assert row["error"] == "" and row["rows_hit"] == row["rows_total"] == 1
    assert (
        "表二五～五｜絞線 8｜同一導線管內或電纜內之載流導線數／7-9／安培容量（A） = 20"
        in prompts[1]
    )
    assert "查表結果" in prompts[1] and "目錄" in prompts[0]
    assert row["duration"] == 1.0


def test_fix_prompt_names_no_real_row_or_table():
    """The first run's example 「800以下」 was copied by qwen3 into 10 C5 lookups; the fix prompt
    gives no example that is a real table id, row or column."""
    from lookup import TABLES
    from prompts import lookup_prompt
    from rules import row_key

    head = lookup_prompt("Q", catalog="", fix=True).split("目錄")[0]
    assert "800以下" in lookup_prompt("Q", catalog="", fix=False)
    for tid, t in TABLES.items():
        assert tid not in head
        assert not any(f"「{row_key(tid, r)}」" in head for r in t["rows"])
        assert not any(f"「{n}」" in head for n in t["names"])
