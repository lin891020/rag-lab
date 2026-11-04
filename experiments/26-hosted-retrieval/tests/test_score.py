"""score.py: 16's ruler (number within 1 %, ladder step, found for abstain), defect questions not
scored, a stopped reply always wrong, and the refusal to score a failed, stale or missing cell."""

import json

import score

Q = {q["id"]: q for q in score.load_questions()}


def _row(cond, qid, content, **kw):
    return {
        "model": "m",
        "condition": cond,
        "qid": qid,
        "chain": Q[qid]["chain"],
        "content": content,
        "error": "",
        "prompt_sha": "a",
        **kw,
    }


def test_defect_questions_are_not_scored():
    defect = [q for q, v in Q.items() if v.get("defect")]
    assert len(defect) == 5
    rows = score.score_rows([_row("cards", q, "{}") for q in defect[:2]], Q)
    assert rows == []


def test_a_stopped_reply_is_wrong_even_on_an_abstain_question():
    q = next(k for k, v in Q.items() if v["expect"]["kind"] == "abstain" and not v.get("defect"))
    r = score.score_rows([_row("cards", q, "", stopped="max_tokens")], Q)[0]
    assert r["right"] is False and r["abstain_ok"] is False and r["grade"] == "stopped"
    ok = score.score_rows(
        [
            _row(
                "cards",
                q,
                '{"found": false, "quote": "", "expression": "", "value": 0, "unit": ""}',
            )
        ],
        Q,
    )[0]
    assert ok["right"] is True


def test_primary_pairs():
    assert {(p[1], p[3]) for p in score.PRIMARY} == {("oracle", "cards"), ("cards", "struct@1500")}
    assert {p[0] for p in score.PRIMARY} == {"claude-haiku-5-5", "claude-opus-5-5"}


def test_score_refuses_a_failed_call_a_stale_row_or_a_missing_cell(tmp_path, monkeypatch):
    qid = next(k for k, v in Q.items() if not v.get("defect"))
    monkeypatch.setattr(score, "RESULTS", tmp_path)
    monkeypatch.setattr(score, "load_questions", lambda: [Q[qid]])
    monkeypatch.setattr(score, "CONDITIONS", ["cards", "oracle"])
    monkeypatch.setattr(score, "MODELS", ["m", "m2"])
    monkeypatch.setattr(score, "RUNS", {"m": ["cards", "oracle"], "m2": ["cards"]})
    monkeypatch.setattr(score, "signatures", lambda qs: {("cards", qid): "a", ("oracle", qid): "a"})
    for rows in (
        [_row("cards", qid, "{}", error="x"), _row("oracle", qid, "{}")],
        [_row("cards", qid, "{}", prompt_sha="b"), _row("oracle", qid, "{}")],
        [_row("cards", qid, "{}")],
    ):
        (tmp_path / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        assert score.main() == 1
    rows = [_row("cards", qid, "{}"), _row("oracle", qid, "{}")]
    (tmp_path / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert score.main() == 1  # m2 has no rows at all: still a missing cell
    rows.append(dict(_row("cards", qid, "{}"), model="m2"))
    (tmp_path / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert score.main() == 0


def test_an_unparseable_reply_is_wrong_even_on_an_abstain_question():
    q = next(k for k, v in Q.items() if v["expect"]["kind"] == "abstain" and not v.get("defect"))
    for content in ("", "not json"):
        r = score.score_rows([_row("cards", q, content)], Q)[0]
        assert r["right"] is False and r["abstain_ok"] is False
