"""score.py: the primary tests are agent − cards and agent − decomp@4000 for each model; 26's
rows are re-scored with this ruler and come out as 26 scored them; scoring refuses while a call
failed, a row is stale or a cell is missing."""

import json

import run
import score


def test_primary_is_agent_against_cards_and_decomp():
    assert score.PRIMARY == [
        (m, "agent", m, b, "all")
        for b in ("cards", "decomp@4000")
        for m in ("claude-haiku-5-5", "claude-opus-5-5")
    ]


def test_26s_rows_score_as_26_scored_them():
    mine = {
        (r["model"], r["condition"], r["qid"]): r
        for r in score.score_rows(score.baseline26(), {q["id"]: q for q in run.load_questions()})
    }
    theirs = [json.loads(ln) for ln in score.SCORED26.open(encoding="utf-8")]
    theirs = [r for r in theirs if r["condition"] in score.BASELINE]
    assert len(mine) == len(theirs) == 2 * 3 * 155
    for r in theirs:
        m = mine[(r["model"], r["condition"], r["qid"])]
        assert (m["right"], m["grade"], m["rows_all_hit"]) == (
            r["right"],
            r["grade"],
            r["rows_all_hit"],
        )


def test_score_refuses_a_failed_stale_or_missing_cell(tmp_path, monkeypatch):
    monkeypatch.setattr(score, "RESULTS", tmp_path)
    qs = run.load_questions()
    good = [
        {
            "model": m,
            "condition": c,
            "qid": q["id"],
            "chain": q["chain"],
            "content": '{"found": false}',
            "error": "",
            "prompt_sha": run.signature(q),
        }
        for m, c, _ in run.grid(qs[:1])
        for q in qs
    ]
    raw = tmp_path / "raw.jsonl"
    for bad in (
        [dict(good[0], error="OSError()")] + good[1:],
        [dict(good[0], prompt_sha="old")] + good[1:],
        good[1:],
    ):
        raw.write_text("".join(json.dumps(r) + "\n" for r in bad))
        assert score.main() == 1
    raw.write_text("".join(json.dumps(r) + "\n" for r in good))
    assert score.main() == 0
