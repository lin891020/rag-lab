"""The whole chain offline: a fake Claude answers the full grid, then score and the report run."""

import csv
import json

import run


def test_chain_offline(tmp_path, monkeypatch):
    import posthoc26
    import score

    def chat(prompt, *, model, system, schema, images=()):
        return {
            "content": '{"found": true, "quote": "", "expression": "8", "value": 8, "unit": "mm²"}',
            "thinking": "",
            "eval_count": 40,
            "prompt_eval_count": 3000,
            "duration": 0.1,
            "done_reason": "end_turn",
            "served_model": model,
            "request_id": "r",
        }

    monkeypatch.setattr(run.claude, "chat", chat)
    monkeypatch.setattr(run.claude, "load_env", lambda *a, **k: None)
    for mod in (score, posthoc26):
        monkeypatch.setattr(mod, "RESULTS", tmp_path)
    assert run.main(["--out", str(tmp_path)]) == 0
    assert score.main() == 0 and posthoc26.main() == 0
    diffs = list(csv.DictReader((tmp_path / "diffs.csv").open(encoding="utf-8")))
    primary = {
        (r["model"], r["a"], r["b"])
        for r in diffs
        if r["primary"] == "True" and r["scope"] == "all"
    }
    assert primary == {
        (m, a, b) for m in run.MODELS for a, b in (("oracle", "cards"), ("cards", "struct@1500"))
    }
    assert all(int(r["n"]) == 127 for r in diffs if r["scope"] == "all" and r["metric"] == "right")
    report = (tmp_path / "report.md").read_text("utf-8")
    assert "gpt-oss:20b（17，同一把尺重算）" in report and "## 雜訊" in report
    meta = json.loads((tmp_path / "meta.json").read_text("utf-8"))
    assert meta["calls"] == 160 * 15 and meta["failed_calls"] == 0
