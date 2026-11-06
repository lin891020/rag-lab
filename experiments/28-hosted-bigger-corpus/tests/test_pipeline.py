"""The whole chain offline: a fake agent searches once or twice and answers the full grid, then
score (with 27's real rows beside) and the report run."""

import csv
import json

import run


class _Searcher:
    units: list = []

    def search(self, query):
        return "表二五～五 PVC 管配線 | 8", {"表二五～五|絞線 8"}


def test_chain_offline(tmp_path, monkeypatch):
    import posthoc28
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
    monkeypatch.setattr(run, "make_searcher", lambda corpus: _Searcher())
    for mod in (score, posthoc28):
        monkeypatch.setattr(mod, "RESULTS", tmp_path)
    assert run.main(["--out", str(tmp_path)]) == 0
    assert score.main() == 0 and posthoc28.main() == 0
    diffs = list(csv.DictReader((tmp_path / "diffs.csv").open(encoding="utf-8")))
    primary = {
        (r["model"], r["a"], r["b"])
        for r in diffs
        if r["primary"] == "True" and r["scope"] == "all" and r["metric"] == "right"
    }
    assert primary == {(m, "agent-big", "agent") for m in run.MODELS}
    assert all(int(r["n"]) == 127 for r in diffs if r["scope"] == "all" and r["metric"] == "right")
    report = (tmp_path / "report.md").read_text("utf-8")
    assert "## 怎麼查的" in report and "## 雜訊：Haiku 兩次" in report
    assert "| agent-big | 155/155 | 155/155 |" in report  # the fake answers the same both times
    assert "## 不叫模型" in report and "## 查到哪幾部法規" in report
    meta = json.loads((tmp_path / "meta.json").read_text("utf-8"))
    assert meta["questions_asked"] == 160 * 6 and meta["failed_questions"] == 0
    assert meta["api_calls"] == 160 * (4 * 2 + 2 * 3)


def test_numbers_in_folds_spacing_and_needs_every_number():
    from posthoc28 import numbers_in

    assert numbers_in("表九三～二：100以下 | - | 8", ["過電流 100以下 | - | 8 | 3.2"])
    assert numbers_in("3以下 130 | 4 117", ["3以下 | 4 | 130 | 117"])  # spaces separate numbers
    assert numbers_in("20,000以下 | 50", ["20000以下 | 50"])
    assert not numbers_in("100以下 | 14", ["100以下 | 8"])
    assert not numbers_in("4117", ["4 117"])
    assert not numbers_in("沒有數字", ["8"])
