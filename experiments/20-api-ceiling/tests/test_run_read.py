"""20 asks Claude the 19 questions: the small models' rows are copied from 19 (not asked), cards
and oracle are re-asked with the exact evidence 19 stored (the prompt rebuilds to the same
length), and one question of each kind runs end to end with a fake Claude client."""

import json

import run_read
from lookup import TABLES
from prompts import lookup_prompt
from rules import row_key
from run_read import (
    COPIED,
    RUNS,
    ask_given,
    ask_lookup,
    catalog,
    copied_rows,
    load_questions,
    parse_lookups,
    price,
    source_evidence,
)

Q = load_questions()
IDS = {q["id"] for q in Q}


def _fake(replies, prompts):
    it = iter(replies)

    def chat(prompt, **kw):
        prompts.append(prompt)
        return {
            "content": next(it),
            "thinking": "",
            "eval_count": 5,
            "prompt_eval_count": 50,
            "duration": 0.5,
            "done_reason": "end_turn",
            "served_model": kw["model"],
        }

    return chat


def test_questions_are_the_three_chains():
    assert len(Q) == 96 and {q["chain"] for q in Q} == {"C1", "C4", "C5"}


def test_every_claude_cell_runs_twice():
    for cells in RUNS.values():
        assert {c.removesuffix("-2") for c in cells} == {"cards", "oracle", "fix"}
        assert len(cells) == 6


def test_copied_rows_are_19s_small_model_cells():
    rows = copied_rows(IDS)
    assert {r["reused_from"] for r in rows} == {"19"}
    assert len(rows) == 96 * sum(len(v) for v in COPIED.values())
    for m, cells in COPIED.items():
        assert {r["condition"] for r in rows if r["model"] == m} == set(cells)


def test_source_evidence_rebuilds_19s_prompts():
    src = source_evidence(IDS)
    assert len(src) == 2 * 96
    for (cell, qid), r in src.items():
        q = next(x for x in Q if x["id"] == qid)
        assert len(run_read.given_prompt(cell, q, r)) == r["prompt_len"], (cell, qid)


def test_parse_lookups_caps_and_fills():
    many = json.dumps({"lookups": [{"table": "表五二", "row": str(i)} for i in range(9)]})
    got, problem = parse_lookups(many)
    assert len(got) == 6 and problem == "over the limit" and got[0]["column"] == ""
    assert parse_lookups("not json") == ([], "unparseable")


def test_ask_lookup_end_to_end(monkeypatch):
    q = next(q for q in Q if q["id"] == "c1-10")  # PVC, 8 mm² 絞線, 60 ℃, 9 conductors -> 20
    prompts: list[str] = []
    replies = [
        '{"lookups": [{"table": "表二五～五", "row": "絞線 8", "column": "9"}]}',
        '{"found": true, "quote": "", "expression": "20", "value": 20, "unit": "A"}',
    ]
    monkeypatch.setattr(run_read.claude, "chat", _fake(replies, prompts))
    row = ask_lookup("claude-opus-5-5", "fix-2", q, catalog())
    assert row["error"] == "" and row["rows_hit"] == row["rows_total"] == 1
    assert "表二五～五｜絞線 8｜" in prompts[1] and "目錄" in prompts[0]
    assert (row["input_tokens"], row["output_tokens"]) == (100, 10)
    assert row["served_model"] == "claude-opus-5-5"


def test_ask_given_keeps_19s_evidence(monkeypatch):
    q = Q[0]
    src = source_evidence({q["id"]})[("cards", q["id"])]
    prompts: list[str] = []
    reply = '{"found": false, "quote": "", "expression": "", "value": 0, "unit": ""}'
    monkeypatch.setattr(run_read.claude, "chat", _fake([reply], prompts))
    row = ask_given("claude-haiku-5-5", "cards-2", q, src)
    assert row["error"] == "" and row["content"] == reply
    assert row["evidence_text"] == src["evidence_text"] and src["evidence_text"] in prompts[0]
    assert (row["rows_hit"], row["rows_total"]) == (src["rows_hit"], src["rows_total"])


def test_ask_given_oracle_rows_lack_table_fields(monkeypatch):
    """19's oracle rows (from 16/17) have no tables_hit: the reply must still be kept."""
    q = Q[0]
    src = source_evidence({q["id"]})[("oracle", q["id"])]
    reply = '{"found": false, "quote": "", "expression": "", "value": 0, "unit": ""}'
    monkeypatch.setattr(run_read.claude, "chat", _fake([reply], []))
    row = ask_given("claude-opus-5-5", "oracle", q, src)
    assert row["error"] == "" and row["content"] == reply and row["date"]


def test_a_refused_call_keeps_its_tokens(monkeypatch):
    """A stop other than end_turn is paid for: its tokens count toward the budget."""

    def refuse(prompt, **kw):
        raise run_read.claude.StopError("refusal", tokens_in=40, tokens_out=3)

    monkeypatch.setattr(run_read.claude, "chat", refuse)
    row = ask_given(
        "claude-opus-5-5", "cards", Q[0], source_evidence({Q[0]["id"]})[("cards", Q[0]["id"])]
    )
    assert row["error"] and (row["input_tokens"], row["output_tokens"]) == (40, 3)


def test_step1_tokens_survive_a_failed_step2(monkeypatch):
    replies = iter(['{"lookups": []}'])

    def chat(prompt, **kw):
        try:
            content = next(replies)
        except StopIteration:
            raise run_read.claude.StopError("max_tokens", tokens_in=7, tokens_out=9) from None
        return {
            "content": content,
            "thinking": "",
            "eval_count": 5,
            "prompt_eval_count": 50,
            "duration": 0.5,
            "done_reason": "end_turn",
        }

    monkeypatch.setattr(run_read.claude, "chat", chat)
    row = ask_lookup("claude-opus-5-5", "fix", Q[0], catalog())
    assert row["error"] and (row["input_tokens"], row["output_tokens"]) == (57, 14)


def test_copied_fix_rows_count_needed_rows_only():
    """19 stored rows_total=2 for c4-16; 20 recounts the copied fix rows from their lookups."""
    rows = [r for r in copied_rows({"c4-16"}) if r["condition"].startswith("fix")]
    assert rows and all(r["rows_total"] == 1 for r in rows)


def test_sample_spreads_over_chains():
    assert {q["chain"] for q in run_read.sample(Q, 3)} == {"C1", "C4", "C5"}


def test_failed_call_is_an_error_row(monkeypatch):
    def boom(prompt, **kw):
        raise ValueError("stop_reason refusal")

    monkeypatch.setattr(run_read.claude, "chat", boom)
    row = ask_lookup("claude-opus-5-5", "fix", Q[0], catalog())
    assert row["error"] and row["content"] == ""


def test_price_per_million_tokens():
    assert price("claude-opus-5-5", 1_000_000, 1_000_000) == 24.0
    assert price("claude-haiku-5-5", 1_000_000, 0) == 0.10


def test_fix_prompt_names_no_real_row_or_table():
    """The first run of 19 copied its example 「800以下」; the fix prompt names no real id."""
    head = lookup_prompt("Q", catalog="", fix=True).split("目錄")[0]
    for tid, t in TABLES.items():
        assert tid not in head
        assert not any(f"「{row_key(tid, r)}」" in head for r in t["rows"])
        assert not any(f"「{n}」" in head for n in t["names"])
