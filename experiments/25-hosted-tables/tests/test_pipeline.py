"""The whole chain offline: a fake Claude answers the full grid, then score, report and posthoc
run on it. Also: clean and glm2 are 24's corpus byte for byte."""

import csv
import json
from pathlib import Path

import pytest
import run

HERE = Path(__file__).parents[1]


def test_clean_and_glm2_are_24s_byte_for_byte():
    for route in ("clean", "glm2"):
        ours = sorted((HERE / "corpus" / route).glob("*.txt"))
        assert len(ours) == 17
        for f in ours:
            theirs = HERE.parent / "24-ocr-header/corpus" / route / f.name
            assert f.read_bytes() == theirs.read_bytes(), (route, f.name)


@pytest.mark.skipif(
    not (run.PNG06.exists() and run.L2.exists()), reason="page images live in data/, not in git"
)
def test_chain_offline(tmp_path, monkeypatch):
    import posthoc25
    import report
    import score

    def chat(prompt, *, model, system, schema, images=()):
        return {
            "content": '{"quote": "", "value": 65, "unit": "A"}',
            "thinking": "",
            "eval_count": 50,
            "prompt_eval_count": 1000 + 4800 * len(images),
            "duration": 0.1,
            "done_reason": "end_turn",
            "served_model": model,
            "request_id": "r",
        }

    monkeypatch.setattr(run.claude, "chat", chat)
    monkeypatch.setattr(run.claude, "load_env", lambda *a, **k: None)
    for mod in (score, report, posthoc25):
        monkeypatch.setattr(mod, "RESULTS", tmp_path)
    assert run.main(["--out", str(tmp_path)]) == 0
    assert score.main() == 0 and report.main() == 0 and posthoc25.main() == 0
    summary = list(csv.DictReader((tmp_path / "summary.csv").open(encoding="utf-8")))
    assert all(float(r["mean"]) == 50 for r in summary if r["metric"] == "eval_count")
    diffs = list(csv.DictReader((tmp_path / "diffs.csv").open(encoding="utf-8")))
    assert {(r["a"], r["b"]) for r in diffs if r["primary"] == "True"} == {
        ("image", "clean"),
        ("glm2", "clean"),
    }
    post = (tmp_path / "posthoc.md").read_text("utf-8")
    assert "## 雜訊：同一格的兩次" in post and "| claude-opus-5-5 | image | 164 | 0 |" in post
    meta = json.loads((tmp_path / "meta.json").read_text("utf-8"))
    assert meta["calls"] == 1312 and meta["failed_calls"] == 0
