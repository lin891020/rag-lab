"""run.py: the grid (agent-big and agent-big-name; Haiku twice, Opus once), the prompt (17's
rules word for word, the evidence line pointing at the six-law library without naming the one
that applies), the resume rule, and one question asked with a fake agent: the queries, the union
of what they returned and which laws it came from are kept, every turn is billed, a stop is
final, a failed call keeps its bill and is asked again."""

import json

import pytest

import claude
import run
import laws
from prompts import _RULES, AGENT_LINE, agent_prompt

Q = run.load_questions()
GOLD = next(q for q in Q if len(q["gold"]["rows"]) == 2 and q["expect"]["kind"] != "abstain")


class _Searcher:
    """Returns the first gold row for any query, the second only for the query "二"."""

    def __init__(self, q):
        self.rows = [f"{t}|{k}" for t, k in q["gold"]["rows"]]

    def search(self, query):
        labels = {self.rows[0]} | ({self.rows[1]} if query == "二" else {"J0030019:25"})
        return f"文字 {query}", labels


def _agent(queries, *, content='{"found": false}', raise_=None):
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


def test_grid_runs_haiku_twice_then_opus_once_on_both_big_corpora():
    g = list(run.grid([{"id": "a"}]))
    assert g == [
        ("claude-haiku-5-5", "agent-big", "a"),
        ("claude-haiku-5-5", "agent-big-2", "a"),
        ("claude-haiku-5-5", "agent-big-name", "a"),
        ("claude-haiku-5-5", "agent-big-name-2", "a"),
        ("claude-opus-5-5", "agent-big", "a"),
        ("claude-opus-5-5", "agent-big-name", "a"),
    ]
    assert [run.corpus_of(c) for _, c, _ in g] == ["big", "big", "big-name", "big-name"] * 1 + [
        "big",
        "big-name",
    ]


def test_prompt_keeps_17s_rules_and_swaps_only_the_evidence_line():
    p = run.prompt_for(Q[0])
    assert _RULES in p and AGENT_LINE in p and p.endswith(f"問題：{Q[0]['question']}")
    assert "<<<" not in p and "不提供原文" not in p
    assert "最多查 5 次" in AGENT_LINE and claude.MAX_SEARCHES == 5
    assert p == agent_prompt(Q[0]["question"])
    for name in ["用戶用電設備裝置規則", *laws.LAWS.values()]:  # the library, every law named
        assert f"《{name}》" in AGENT_LINE and f"《{name}》" in run.TOOL["description"]
    assert "查《用戶用電設備裝置規則》的條文" not in AGENT_LINE  # not told which law applies


def test_signature_covers_the_corpus_the_tool_the_caps_and_the_search(monkeypatch):
    assert run.signature(Q[0], "agent-big") == run.signature(Q[0], "agent-big-2")
    assert run.signature(Q[0], "agent-big") != run.signature(Q[0], "agent-big-name")
    sigs = [run.signature(Q[0], "agent-big")]
    for mod, name, value in (  # the modules run.py holds: another test may have re-imported them
        (run, "TOOL", {**run.TOOL, "description": "x"}),
        (run.claude, "MAX_SEARCHES", 3),
        (run.claude, "EFFORT", "high"),
        (run.claude, "MAX_TOKENS", 1),
        (run.search, "BUDGET", 4000),
        (run.search, "BGE_DIGEST", "x"),
        (run, "index_sha", lambda corpus: "x"),
    ):
        monkeypatch.setattr(mod, name, value)
        sigs.append(run.signature(Q[0], "agent-big"))
    assert len(set(sigs)) == len(sigs)


def test_index_sha_follows_the_built_index(monkeypatch):
    a = run.index_sha("big")
    assert a != run.index_sha("big-name")
    run.index_sha.cache_clear()
    units, blocks = run.search.build_units("big")
    monkeypatch.setattr(run.search, "build_units", lambda c: (units[:-1], blocks))
    assert run.index_sha("big") != a
    run.index_sha.cache_clear()


def test_ask_keeps_queries_the_union_of_hits_and_the_laws_it_came_from(monkeypatch):
    monkeypatch.setattr(claude, "agent", _agent(["一", "二"]))
    row = run.ask("claude-opus-5-5", "agent-big", GOLD, _Searcher(GOLD))
    assert row["laws_seen"] == ["用戶用電設備裝置規則", "輸配電設備裝置規則"]
    assert row["added_labels"] == 1
    assert row["queries"] == ["一", "二"] and row["n_searches"] == 2
    assert (row["rows_hit"], row["rows_total"]) == (2, 2)
    assert row["evidence_chars"] == len("文字 一") + len("文字 二")
    assert (row["input_tokens"], row["cache_write_tokens"], row["cache_read_tokens"]) == (
        2000,
        200,
        1000,
    )
    assert row["output_tokens"] == 100 and row["api_calls"] == 2 and row["error"] == ""
    assert row["prompt_sha"] == run.signature(GOLD, "agent-big")


def test_a_stop_is_final_and_keeps_queries_and_bill(monkeypatch):
    stop = claude.StopError(
        "cap", reason="max_turns", turns=8, refused=2, request_ids=["a"], served_model="m"
    )
    monkeypatch.setattr(claude, "agent", _agent(["一"], raise_=stop))
    row = run.ask("claude-opus-5-5", "agent-big", GOLD, _Searcher(GOLD))
    assert row["stopped"] == "max_turns" and row["error"] == "" and row["content"] == ""
    assert row["queries"] == ["一"] and row["rows_hit"] == 1 and row["input_tokens"] == 1000
    assert (row["refused_searches"], row["request_ids"], row["served_model"]) == (2, ["a"], "m")


def test_a_failed_call_keeps_its_bill_and_is_an_error(monkeypatch):
    monkeypatch.setattr(claude, "agent", _agent(["一"], raise_=OSError("net")))
    row = run.ask("claude-opus-5-5", "agent-big", GOLD, _Searcher(GOLD))
    assert row["error"] and row["input_tokens"] == 1000 and row["api_calls"] == 1


def test_spent_counts_every_row_with_cache_tokens():
    r = {
        "model": "claude-opus-5-5",
        "input_tokens": 1_000_000,
        "cache_write_tokens": 1_000_000,
        "cache_read_tokens": 1_000_000,
        "output_tokens": 0,
    }
    assert run.spent_usd([r, {**r, "error": "x"}]) == pytest.approx(2 * 9.20)


def test_done_needs_a_final_row_with_the_same_prompt(tmp_path):
    p = tmp_path / "raw.jsonl"
    base = {"model": "m", "condition": "agent", "qid": "q", "error": "", "prompt_sha": "a"}
    rows = [dict(base, error="OSError()"), dict(base), dict(base, qid="r", prompt_sha="old")]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    sig = {("agent", "q"): "a", ("agent", "r"): "new"}
    assert run.done_keys(p, sig) == {("m", "agent", "q")}


def test_main_stops_once_the_budget_is_passed(tmp_path, monkeypatch):
    monkeypatch.setattr(claude, "agent", _agent(["一"]))
    monkeypatch.setattr(claude, "load_env", lambda *a, **k: None)
    monkeypatch.setattr(run, "make_searcher", lambda corpus: _Searcher(GOLD))
    assert run.main(["--out", str(tmp_path), "--budget", "0.000001", "--sample", "2"]) == 2
    assert len(run.read_raw(tmp_path / "raw.jsonl")) == 1


def test_main_aborts_after_five_failures_in_a_row(tmp_path, monkeypatch):
    monkeypatch.setattr(claude, "agent", _agent([], raise_=OSError("400 every time")))
    monkeypatch.setattr(claude, "load_env", lambda *a, **k: None)
    monkeypatch.setattr(run, "make_searcher", lambda corpus: _Searcher(GOLD))
    assert run.main(["--out", str(tmp_path), "--sample", "10"]) == 3
    assert len(run.read_raw(tmp_path / "raw.jsonl")) == run.MAX_FAILS_IN_A_ROW == 5
