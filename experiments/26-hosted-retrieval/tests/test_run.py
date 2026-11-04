"""run.py: the grid (Haiku twice, Opus once), 17's stored evidence and prompt reused verbatim,
the resume rule, and an offline run with a fake Claude: every call is billed, a stop is final,
a failed call is asked again, and the run stops once the cost passes the budget."""

import json

import run
from prompts import user_prompt

Q = run.load_questions()


def test_grid_runs_haiku_twice_then_opus_once():
    g = list(run.grid([{"id": "a"}]))
    assert [m for m, _, _ in g] == ["claude-haiku-5-5"] * 10 + ["claude-opus-5-5"] * 5
    assert {c for m, c, _ in g if m == "claude-opus-5-5"} == set(run.BASE)
    assert run.BASE == ["closed", "struct@1500", "cards", "decomp@4000", "oracle"]


def test_prompts_are_the_ones_17_sent():
    sent = run.evidence17()
    corpus_checked = 0
    for q in Q:
        for c in run.BASE[1:]:
            src = sent[(c, q["id"])]
            assert len(run.prompt_for(c, q, sent)) == src["prompt_len"], (c, q["id"])
            corpus_checked += 1
    assert corpus_checked == 4 * 160


def test_closed_book_says_no_evidence():
    p = run.prompt_for("closed", Q[0], {})
    assert p == user_prompt(Q[0]["question"], evidence=None, label="", tool=False)
    assert "不提供原文" in p


def test_done_needs_a_final_row_with_the_same_prompt(tmp_path):
    p = tmp_path / "raw.jsonl"
    base = {"model": "m", "qid": "q", "content": "{}", "error": "", "prompt_sha": "a"}
    rows = [
        dict(base, condition="cards"),
        dict(base, condition="oracle", prompt_sha="old"),
        dict(base, condition="closed", error="x"),
        dict(base, condition="decomp@4000", content="", stopped="max_tokens"),
    ]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    sig = {(c, "q"): "a" for c in ("cards", "oracle", "closed", "decomp@4000")}
    assert run.done_keys(p, sig) == {("m", "cards", "q"), ("m", "decomp@4000", "q")}


def _fake_chat(transient=(), stop=()):
    calls = []

    def chat(prompt, *, model, system, schema, images=()):
        calls.append(model)
        if len(calls) in transient:
            raise OSError("connection reset")
        if len(calls) in stop:
            raise run.claude.StopError(
                "stop_reason max_tokens", tokens_in=1000, tokens_out=16000, reason="max_tokens"
            )
        return {
            "content": '{"found": true, "quote": "", "expression": "1", "value": 1, "unit": "A"}',
            "thinking": "",
            "eval_count": 100,
            "prompt_eval_count": 1000,
            "duration": 0.1,
            "done_reason": "end_turn",
            "served_model": model,
            "request_id": "r",
        }

    return chat, calls


def test_offline_run_bills_every_call_and_stops_at_the_budget(tmp_path, monkeypatch):
    chat, calls = _fake_chat(transient=(2,), stop=(3,))
    monkeypatch.setattr(run.claude, "chat", chat)
    monkeypatch.setattr(run.claude, "load_env", lambda *a, **k: None)
    rc = run.main(["--sample", "2", "--out", str(tmp_path), "--budget", "1"])
    rows = [json.loads(ln) for ln in (tmp_path / "raw.jsonl").read_text("utf-8").splitlines()]
    assert rc == 0 and len(rows) == 30  # 2 questions x (10 Haiku + 5 Opus)
    assert (
        sum(bool(r["error"]) for r in rows) == 1 and sum(bool(r.get("stopped")) for r in rows) == 1
    )
    stopped = next(r for r in rows if r.get("stopped"))
    assert (stopped["input_tokens"], stopped["output_tokens"]) == (1000, 16000)
    assert all(r["eval_count"] == r["output_tokens"] for r in rows if not r["error"])
    assert all("rows_total" in r for r in rows if not r["condition"].startswith("closed"))
    meta = json.loads((tmp_path / "meta.json").read_text("utf-8"))
    assert (meta["failed_calls"], meta["stopped"]) == (1, 1) and meta["spent_usd"] > 0
    n = len(calls)
    assert run.main(["--sample", "2", "--out", str(tmp_path), "--budget", "1"]) == 0
    assert len(calls) == n + 1  # only the failed call is asked again
    assert run.main(["--sample", "2", "--out", str(tmp_path / "b"), "--budget", "0.001"]) == 2
