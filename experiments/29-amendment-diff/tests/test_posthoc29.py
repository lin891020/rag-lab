"""posthoc29.drift_probe: each model's drift rows are compared with 27's rows of the same model
(Haiku and Opus rows share qids, so the model must be part of the key), and 27's own two Haiku
runs on the same questions are reported as the noise reference."""

import json

import posthoc29
from run import HAIKU, OPUS


def _row(model, cond, qid, found, value):
    return {
        "model": model,
        "condition": cond,
        "qid": qid,
        "content": json.dumps({"found": found, "value": value}),
        "error": "",
    }


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_drift_probe_compares_each_model_with_its_own_27_rows(tmp_path, monkeypatch):
    drift = tmp_path / "drift"
    drift.mkdir()
    now = [
        _row(HAIKU, "agent", "q1", True, 1),
        _row(HAIKU, "agent", "q2", True, 2),
        _row(OPUS, "agent", "q1", True, 9),
        _row(OPUS, "agent", "q2", True, 2),
    ]
    then = [
        _row(HAIKU, "agent", "q1", True, 1),
        _row(HAIKU, "agent", "q2", True, 5),
        _row(HAIKU, "agent-2", "q1", True, 1),
        _row(HAIKU, "agent-2", "q2", True, 2),
        _row(OPUS, "agent", "q1", True, 9),
        _row(OPUS, "agent", "q2", True, 2),
    ]
    _write(drift / "raw.jsonl", now)
    raw27 = tmp_path / "raw27.jsonl"
    _write(raw27, then)
    monkeypatch.setattr(posthoc29, "DRIFT", drift)
    monkeypatch.setattr(posthoc29, "RAW27", raw27)
    text = "\n".join(posthoc29.drift_probe())
    assert "Haiku" in text and "同答 1/2" in text  # q1 same, q2 differs (2 vs 5)
    assert "Opus" in text and "同答 2/2" in text
    assert "27 自己兩次" in text and "1/2" in text  # 27's agent vs agent-2: q1 same, q2 differs


def test_drift_probe_says_so_when_nothing_ran(tmp_path, monkeypatch):
    monkeypatch.setattr(posthoc29, "DRIFT", tmp_path / "none")
    assert posthoc29.drift_probe() == ["（沒跑）"]
