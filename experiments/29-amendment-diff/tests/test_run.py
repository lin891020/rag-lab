"""run.py: the grid (closed / agent / agent-2 / oracle for Haiku, closed / agent / oracle for
Opus; oracle skips T1), the prompt per condition (the tool line only in agent, the old article
only in oracle, one instruction in closed), the signature, and one question asked with a fake
agent or chat: queries, the union of labels, old_hit, every token billed, a stop is final, a
failed call keeps its bill."""

import json

import pytest

import claude
import run
from prompts import LINES, TOOL

Q = run.load_questions()
T1 = next(q for q in Q if q["kind"] == "T1" and q["old_num"])
T1_NEW = next(q for q in Q if q["kind"] == "T1" and q["old_num"] is None)
T2 = next(q for q in Q if q["kind"] == "T2")
T3 = next(q for q in Q if q["kind"] == "T3")


class _Searcher:
    """Returns the gold old article for any query, plus a neighbour."""

    def __init__(self, q):
        self.gold = q["gold_labels"]
        self.cache = type("C", (), {"save": lambda self: None})()

    def search(self, query):
        return f"文字 {query}", set(self.gold) | {"old:999"}


def _agent(queries, *, content='{"found": false, "old_article": "", "quote": ""}', raise_=None):
    def agent(prompt, *, model, system, schema, tool, run_tool, usage, **kw):
        for qq in queries:
            run_tool(qq)
            for k, v in (
                ("input", 1000),
                ("cache_write", 100),
                ("cache_read", 500),
                ("output", 50),
            ):
                usage[k] += v
            usage["calls"] += 1
        if raise_ is not None:
            raise raise_
        return {
            "content": content,
            "turns": len(queries) + 1,
            "searches": len(queries),
            "refused": 0,
            "duration": 1.0,
            "done_reason": "end_turn",
            "served_model": model,
            "request_ids": ["r"],
        }

    return agent


def _chat(content='{"found": true}'):
    def chat(prompt, *, model, system, schema, usage):
        usage["input"] += 700
        usage["output"] += 30
        usage["calls"] += 1
        return {
            "content": content,
            "turns": 1,
            "searches": 0,
            "refused": 0,
            "duration": 0.5,
            "done_reason": "end_turn",
            "served_model": model,
            "request_ids": ["c"],
        }

    return chat


def test_grid_has_the_cells_the_plan_names_and_oracle_skips_t1():
    g = list(run.grid([T1, T2]))
    assert ("claude-haiku-5-5", "agent-2", T1["id"]) in g
    assert ("claude-opus-5-5", "oracle", T2["id"]) in g
    assert ("claude-opus-5-5", "oracle", T1["id"]) not in g
    assert ("claude-opus-5-5", "agent-2", T1["id"]) not in g
    assert len(g) == 7 + 5  # Haiku 4 conds × 2 q − oracle T1, Opus 3 × 2 − oracle T1
    assert run.RUNS == {
        "claude-haiku-5-5": ["closed", "agent", "agent-2", "oracle"],
        "claude-opus-5-5": ["closed", "agent", "oracle"],
    }


def test_prompt_differs_by_condition_in_the_evidence_only():
    pa, po, pc = run.prompt(T2, "agent"), run.prompt(T2, "oracle"), run.prompt(T2, "closed")
    assert LINES["agent"] in pa and "search_old" in pa and "search_old" not in pc
    assert LINES["oracle"] in po and run.old_texts()[T2["old_num"]] in po
    assert LINES["closed"] in pc and "不要用你自己知道" not in pc  # one instruction, not two
    for p in (pa, po, pc):
        assert p.endswith(T2["question"]) and T2["new_text"] in p
    assert run.prompt(T2, "agent-2") == pa
    with pytest.raises(KeyError):
        run.prompt(T1_NEW, "oracle")  # no old article: oracle never asks T1


def test_schema_and_rules_follow_the_kind():
    assert "changed" in run.prompt(T2, "agent") and "changed" not in run.prompt(T1, "agent")
    assert "old_value" in run.prompt(T3, "agent") and "old_value" not in run.prompt(T2, "agent")


def test_signature_covers_prompt_tool_caps_and_index(monkeypatch):
    assert run.signature(T1, "agent") == run.signature(T1, "agent-2")
    assert run.signature(T1, "agent") != run.signature(T1, "closed")
    assert run.signature(T2, "oracle") != run.signature(T2, "closed")
    sigs = [run.signature(T1, "agent")]
    for mod, name, value in (
        (run, "TOOL", {**TOOL, "description": "x"}),
        (run.claude, "MAX_SEARCHES", 3),
        (run.claude, "EFFORT", "high"),
        (run.search, "BUDGET", 4000),
        (run.search, "index_sha", lambda: "x"),
    ):
        monkeypatch.setattr(mod, name, value)
        sigs.append(run.signature(T1, "agent"))
    assert len(set(sigs)) == len(sigs)
    closed = run.signature(T1, "closed")
    monkeypatch.setattr(run.search, "index_sha", lambda: "y")
    assert run.signature(T1, "closed") == closed  # no index in closed's signature


def test_ask_agent_keeps_queries_labels_old_hit_and_the_bill(monkeypatch):
    monkeypatch.setattr(claude, "agent", _agent(["一", "二"]))
    row = run.ask("claude-opus-5-5", "agent", T1, _Searcher(T1))
    assert row["queries"] == ["一", "二"] and row["n_searches"] == 2
    assert row["old_hit"] is True and set(row["labels"]) == {*T1["gold_labels"], "old:999"}
    assert (row["input_tokens"], row["cache_write_tokens"], row["cache_read_tokens"]) == (
        2000,
        200,
        1000,
    )
    assert row["output_tokens"] == 100 and row["api_calls"] == 2 and row["error"] == ""
    assert row["prompt_sha"] == run.signature(T1, "agent") and row["kind"] == "T1"


def test_an_added_question_has_no_old_hit(monkeypatch):
    monkeypatch.setattr(claude, "agent", _agent(["一"]))
    row = run.ask("claude-opus-5-5", "agent", T1_NEW, _Searcher(T1))
    assert row["old_hit"] is None and row["labels"]


def test_ask_closed_and_oracle_use_chat_and_have_no_searches(monkeypatch):
    monkeypatch.setattr(claude, "chat", _chat())
    monkeypatch.setattr(claude, "agent", _agent(["x"]))  # must not be called
    for cond in ("closed", "oracle"):
        row = run.ask("claude-haiku-5-5", cond, T2, None)
        assert row["queries"] == [] and row["old_hit"] is None and row["api_calls"] == 1
        assert row["input_tokens"] == 700 and row["content"] == '{"found": true}'


def test_a_stop_is_final_and_keeps_queries_and_bill(monkeypatch):
    stop = claude.StopError(
        "cap", reason="max_turns", turns=8, refused=2, request_ids=["a"], served_model="m"
    )
    monkeypatch.setattr(claude, "agent", _agent(["一"], raise_=stop))
    row = run.ask("claude-opus-5-5", "agent", T1, _Searcher(T1))
    assert row["stopped"] == "max_turns" and row["error"] == "" and row["content"] == ""
    assert row["queries"] == ["一"] and row["old_hit"] is True and row["input_tokens"] == 1000


def test_a_failed_call_keeps_its_bill_and_is_an_error(monkeypatch):
    monkeypatch.setattr(claude, "agent", _agent(["一"], raise_=OSError("net")))
    row = run.ask("claude-opus-5-5", "agent", T1, _Searcher(T1))
    assert row["error"] and row["input_tokens"] == 1000 and row["api_calls"] == 1


def test_done_needs_a_final_row_with_the_same_prompt(tmp_path):
    p = tmp_path / "raw.jsonl"
    base = {"model": "m", "condition": "agent", "qid": "q", "error": "", "prompt_sha": "a"}
    rows = [dict(base, error="OSError()"), dict(base), dict(base, qid="r", prompt_sha="old")]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert run.done_keys(p, {("agent", "q"): "a", ("agent", "r"): "new"}) == {("m", "agent", "q")}


def test_per_kind_takes_n_of_each_kind():
    s = run.per_kind(Q, 3)
    assert [q["kind"] for q in s].count("T1") == 3 and len(s) == 9


def test_kind_filter_keeps_one_kind(tmp_path, monkeypatch):
    monkeypatch.setattr(claude, "agent", _agent(["一"]))
    monkeypatch.setattr(claude, "chat", _chat())
    monkeypatch.setattr(claude, "load_env", lambda *a, **k: None)
    monkeypatch.setattr(run, "make_searcher", lambda: _Searcher(T2))
    assert run.main(["--out", str(tmp_path), "--kind", "T3", "--per-kind", "2"]) == 0
    rows = run.read_raw(tmp_path / "raw.jsonl")
    assert {r["kind"] for r in rows} == {"T3"} and len(rows) == 2 * 7  # 7 cells, oracle asks T3


def test_main_stops_once_the_budget_is_passed(tmp_path, monkeypatch):
    monkeypatch.setattr(claude, "agent", _agent(["一"]))
    monkeypatch.setattr(claude, "chat", _chat())
    monkeypatch.setattr(claude, "load_env", lambda *a, **k: None)
    monkeypatch.setattr(run, "make_searcher", lambda: _Searcher(T1))
    assert run.main(["--out", str(tmp_path), "--budget", "0.000001", "--sample", "2"]) == 2
    assert len(run.read_raw(tmp_path / "raw.jsonl")) == 1


def test_main_aborts_after_five_failures_in_a_row(tmp_path, monkeypatch):
    monkeypatch.setattr(claude, "chat", _chat())

    def boom(*a, **k):
        raise OSError("400 every time")

    monkeypatch.setattr(claude, "chat", boom)
    monkeypatch.setattr(claude, "load_env", lambda *a, **k: None)
    monkeypatch.setattr(run, "make_searcher", lambda: _Searcher(T1))
    assert run.main(["--out", str(tmp_path), "--sample", "10"]) == 3
    assert len(run.read_raw(tmp_path / "raw.jsonl")) == run.MAX_FAILS_IN_A_ROW == 5
