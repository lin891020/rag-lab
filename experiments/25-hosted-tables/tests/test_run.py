"""run.py: the grid (cheap model first, each condition twice), the pages each question sends, the
resume rule, and an offline run with a fake Claude: every call is billed, failed ones too, and the
run stops once the cost passes the budget."""

import json

import pytest
import run

needs_pages = pytest.mark.skipif(
    not (run.PNG06.exists() and run.L2.exists()), reason="page images live in data/, not in git"
)


def test_grid_runs_haiku_first_and_each_condition_twice():
    g = list(run.grid(run.MODELS, [{"id": "a"}]))
    assert run.MODELS[0] == "claude-haiku-5-5" and len(g) == 2 * 8
    assert {c.removesuffix("-2") for _, c, _ in g} == {"clean", "glm2", "image", "image-L2"}
    assert [m for m, _, _ in g][:8] == ["claude-haiku-5-5"] * 8


@needs_pages
def test_pages_in_order_and_l2_matches_22s_manifest():
    assert [p.name for p in run.pages_for("image", "表五六")] == ["表五六-p0.png", "表五六-p1.png"]
    assert [p.name for p in run.pages_for("image-L2", "表五六")] == [
        "表五六-p0.jpg",
        "表五六-p1.jpg",
    ]
    shas = run.check_images(run.load_questions())
    assert len(shas) == 36  # 18 pages from each source


def test_done_needs_a_final_row_with_the_same_prompt_and_images(tmp_path):
    p = tmp_path / "raw.jsonl"
    base = {
        "model": "m",
        "qid": "q",
        "content": "{}",
        "error": "",
        "prompt_sha": "a",
        "images": ["s"],
    }
    rows = [
        dict(base, condition="image"),
        dict(base, condition="image-2", images=["old"]),
        dict(base, condition="clean", images=[], error="x"),
        dict(base, condition="glm2", images=[], content="", stopped="max_tokens"),
    ]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    sig = {
        ("image", "q"): ("a", ["s"]),
        ("image-2", "q"): ("a", ["s"]),
        ("clean", "q"): ("a", []),
        ("glm2", "q"): ("a", []),
    }
    assert run.done_keys(p, sig) == {("m", "image", "q"), ("m", "glm2", "q")}  # a stop is final


def _fake_chat(transient=(), stop=()):
    calls = []

    def chat(prompt, *, model, system, schema, images=()):
        calls.append((model, len(images)))
        if len(calls) in transient:
            raise OSError("connection reset")
        if len(calls) in stop:
            raise run.claude.StopError(
                "stop_reason max_tokens", tokens_in=1000, tokens_out=16000, reason="max_tokens"
            )
        return {
            "content": '{"quote": "", "value": 1, "unit": "A"}',
            "thinking": "",
            "eval_count": 100,
            "prompt_eval_count": 1000,
            "duration": 0.1,
            "done_reason": "end_turn",
            "served_model": model,
            "request_id": "r",
        }

    return chat, calls


@needs_pages
def test_offline_run_bills_every_call_and_stops_at_the_budget(tmp_path, monkeypatch):
    chat, calls = _fake_chat(transient=(2,), stop=(3,))
    monkeypatch.setattr(run.claude, "chat", chat)
    monkeypatch.setattr(run.claude, "load_env", lambda *a, **k: None)
    rc = run.main(["--sample", "2", "--out", str(tmp_path), "--budget", "1"])
    rows = [json.loads(ln) for ln in (tmp_path / "raw.jsonl").read_text("utf-8").splitlines()]
    assert rc == 0 and len(rows) == 32
    failed = [r for r in rows if r["error"]]
    stopped = [r for r in rows if r.get("stopped")]
    assert len(failed) == 1 and len(stopped) == 1
    assert (stopped[0]["input_tokens"], stopped[0]["output_tokens"]) == (1000, 16000)
    assert stopped[0]["error"] == "" and stopped[0]["content"] == ""
    assert all(r["eval_count"] == r["output_tokens"] for r in rows if not r["error"])
    assert all(r["images"] for r in rows if r["condition"].startswith("image"))
    assert not any(r["images"] for r in rows if not r["condition"].startswith("image"))
    meta = json.loads((tmp_path / "meta.json").read_text("utf-8"))
    assert (meta["failed_calls"], meta["stopped"]) == (1, 1) and meta["spent_usd"] > 0
    n = len(calls)
    assert run.main(["--sample", "2", "--out", str(tmp_path), "--budget", "1"]) == 0
    assert len(calls) == n + 1  # only the transient failure is asked again
    meta = json.loads((tmp_path / "meta.json").read_text("utf-8"))
    assert meta["failed_calls"] == 1  # every failed call ever made, the retried one too
    assert run.main(["--sample", "2", "--out", str(tmp_path / "b"), "--budget", "0.001"]) == 2


@needs_pages
def test_pages_are_checked_against_the_hashes_git_records(monkeypatch):
    qs = run.load_questions()
    run.check_images(qs)
    real = run.recorded_hashes()
    fake = dict(real, **{"image/表五六-p1.png": "0" * 64})
    monkeypatch.setattr(run, "recorded_hashes", lambda: fake)
    with pytest.raises(SystemExit):
        run.check_images(qs)
