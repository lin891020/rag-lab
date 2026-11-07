"""The whole chain offline: a fake agent searches once and answers, a fake chat answers, the full
grid is written, then score and the report run; the primary pairs and the report's sections are
what the plan names."""

import csv
import json

import run


class _Searcher:
    cache = type("C", (), {"save": lambda self: None})()

    def search(self, query):
        return "第 16 條 低壓導線", {"old:16", "old:13-1"}


def _reply(kind, q):
    if kind == "T1":
        return json.dumps(
            {"found": q["old_num"] is not None, "old_article": q["old_num"] or "", "quote": ""},
            ensure_ascii=False,
        )
    if kind == "T2":
        return json.dumps(
            {
                "found": True,
                "old_article": q["old_num"],
                "changed": q["expect"]["changed"],
                "quote_new": "",
                "quote_old": "",
            },
            ensure_ascii=False,
        )
    e = q["expect"]
    return json.dumps(
        {
            "found": True,
            "old_article": q["old_num"],
            "old_value": e["old_value"],
            "new_value": e["new_value"],
            "unit": "",
            "quote_old": "",
            "quote_new": "",
        },
        ensure_ascii=False,
    )


def test_chain_offline(tmp_path, monkeypatch):
    import posthoc29
    import score

    byq = {q["question"]: q for q in run.load_questions()}

    def which(prompt):
        return next(q for text, q in byq.items() if prompt.endswith(text))

    def agent(prompt, *, model, system, schema, tool, run_tool, usage, **kw):
        q = which(prompt)
        run_tool("查")
        usage.update(input=3000, cache_write=500, cache_read=2000, output=60, calls=2)
        return {
            "content": _reply(q["kind"], q),
            "turns": 2,
            "searches": 1,
            "refused": 0,
            "duration": 0.1,
            "done_reason": "end_turn",
            "served_model": model,
            "request_ids": ["r"],
        }

    def chat(prompt, *, model, system, schema, usage):
        q = which(prompt)
        usage.update(input=1500, output=40, calls=1)
        return {
            "content": _reply(q["kind"], q),
            "turns": 1,
            "searches": 0,
            "refused": 0,
            "duration": 0.1,
            "done_reason": "end_turn",
            "served_model": model,
            "request_ids": ["c"],
        }

    monkeypatch.setattr(run.claude, "agent", agent)
    monkeypatch.setattr(run.claude, "chat", chat)
    monkeypatch.setattr(run.claude, "load_env", lambda *a, **k: None)
    monkeypatch.setattr(run, "make_searcher", lambda: _Searcher())
    for mod in (score, posthoc29):
        monkeypatch.setattr(mod, "RESULTS", tmp_path)
    assert run.main(["--out", str(tmp_path)]) == 0
    assert score.main() == 0 and posthoc29.main() == 0
    diffs = list(csv.DictReader((tmp_path / "diffs.csv").open(encoding="utf-8")))
    primary = {
        (r["model"], r["a"], r["b"], r["scope"])
        for r in diffs
        if r["primary"] == "True" and r["metric"] == "right"
    }
    assert primary == {(m, "oracle", "agent", "T2+T3") for m in run.MODELS}
    n_all = len(byq)
    n_t1 = sum(q["kind"] == "T1" for q in byq.values())
    assert all(
        int(r["n"]) == n_all - n_t1
        for r in diffs
        if r["scope"] == "T2+T3" and r["metric"] == "right"
    )
    report = (tmp_path / "report.md").read_text("utf-8")
    for section in (
        "## 舊版正解撈全率",
        "## T1：答出的舊條號",
        "## 怎麼查的",
        "## 雜訊：Haiku agent 兩次",
        "## 漂移探針",
        "## 花費",
        "## 答錯的題",
    ):
        assert section in report
    assert f"| all | {n_all}/{n_all} |" in report  # the fake answers the same both times
    meta = json.loads((tmp_path / "meta.json").read_text("utf-8"))
    assert (
        meta["questions_asked"] == n_all * 7 - 2 * n_t1 and meta["failed_questions"] == 0
    )  # oracle skips T1 (two models)
    assert (
        meta["api_calls"] == (n_all * 3) * 2 + (n_all - n_t1) * 2 * 1 + n_all * 2 * 1
    )  # agent cells ×2 calls, oracle ×1, closed ×1
