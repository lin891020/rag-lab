"""The whole chain offline: a fake agent searches once or twice and answers the full grid, then
score (with 26's real rows beside) and the report run."""

import csv
import json

import run


class _Searcher:
    def search(self, query):
        return "表二五～五 PVC 管配線 | 8", {"表二五～五|絞線 8"}


def test_chain_offline(tmp_path, monkeypatch):
    import posthoc27
    import score

    def agent(prompt, *, model, system, schema, tool, run_tool, usage, **kw):
        n = 1 if model == run.HAIKU else 2
        for i in range(n):
            run_tool(f"查 {i}")
        usage.update(input=3000, cache_write=500, cache_read=2000, output=60, calls=n + 1)
        return {
            "content": '{"found": true, "quote": "PVC 管配線", "expression": "8", "value": 8, '
            '"unit": "mm²"}',
            "turns": n + 1,
            "searches": n,
            "refused": 0,
            "duration": 0.1,
            "done_reason": "end_turn",
            "served_model": model,
            "request_ids": ["r"],
        }

    monkeypatch.setattr(run.claude, "agent", agent)
    monkeypatch.setattr(run.claude, "load_env", lambda *a, **k: None)
    monkeypatch.setattr(run, "make_searcher", lambda: _Searcher())
    for mod in (score, posthoc27):
        monkeypatch.setattr(mod, "RESULTS", tmp_path)
    assert run.main(["--out", str(tmp_path)]) == 0
    assert score.main() == 0 and posthoc27.main() == 0
    diffs = list(csv.DictReader((tmp_path / "diffs.csv").open(encoding="utf-8")))
    primary = {
        (r["model"], r["a"], r["b"])
        for r in diffs
        if r["primary"] == "True" and r["scope"] == "all" and r["metric"] == "right"
    }
    assert primary == {(m, "agent", b) for m in run.MODELS for b in ("cards", "decomp@4000")}
    assert all(int(r["n"]) == 127 for r in diffs if r["scope"] == "all" and r["metric"] == "right")
    report = (tmp_path / "report.md").read_text("utf-8")
    assert "## 怎麼查的" in report and "## 雜訊：Haiku 兩次" in report
    assert "| 155/155 | 155/155 |" in report  # the fake answers the same both times
    meta = json.loads((tmp_path / "meta.json").read_text("utf-8"))
    assert meta["questions_asked"] == 160 * 3 and meta["failed_questions"] == 0
    assert meta["api_calls"] == 160 * (2 + 2 + 3)
