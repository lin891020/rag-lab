"""score.py: one ruler per kind (T1 number or 新增, T2 boolean, T3 both values within 1 %, 0 exact,
swapped flagged), stopped or unparsable is wrong, answer_grounded only on agent rows, the primary
pairs are oracle − agent on T2+T3 per model, and scoring refuses a failed, stale or missing cell."""

import json

import run
import score

Q = {q["id"]: q for q in run.load_questions()}
T1 = next(q for q in Q.values() if q["kind"] == "T1" and q["old_num"])
T1_NEW = next(q for q in Q.values() if q["kind"] == "T1" and q["old_num"] is None)
T2_SAME = next(q for q in Q.values() if q["kind"] == "T2" and not q["expect"]["changed"])
T2_CHG = next(q for q in Q.values() if q["kind"] == "T2" and q["expect"]["changed"])
T3 = next(q for q in Q.values() if q["kind"] == "T3")


def _j(obj, q):
    return score.judge(json.dumps(obj, ensure_ascii=False), q)


def test_t1_right_when_the_number_matches_in_any_spelling_or_new_is_found_false():
    n = T1["old_num"]
    for spelled in (n, f"第 {n} 條", f"第{n}條"):
        assert _j({"found": True, "old_article": spelled, "quote": ""}, T1)["right"]
    assert not _j({"found": True, "old_article": "1", "quote": ""}, T1)["right"] or n == "1"
    assert not _j({"found": False, "old_article": "", "quote": ""}, T1)["right"]
    assert _j({"found": False, "old_article": "", "quote": ""}, T1_NEW)["right"]
    assert not _j({"found": True, "old_article": "5", "quote": ""}, T1_NEW)["right"]


def test_t1_chinese_numeral_spelling_folds():
    q = dict(T1, expect={"old_num": "13-1"})
    assert _j({"found": True, "old_article": "第十三條之一", "quote": ""}, q)["right"]
    assert _j({"found": True, "old_article": "13-1", "quote": ""}, q)["right"]


def test_t2_right_iff_changed_matches_and_found():
    base = {"found": True, "old_article": "x", "quote_new": "", "quote_old": ""}
    assert _j({**base, "changed": False}, T2_SAME)["right"]
    assert not _j({**base, "changed": True}, T2_SAME)["right"]
    assert _j({**base, "changed": True}, T2_CHG)["right"]
    assert not _j({**base, "found": False, "changed": True}, T2_CHG)["right"]


def test_t3_tolerance_zero_and_swapped():
    e = T3["expect"]
    base = {"found": True, "old_article": "x", "unit": "", "quote_old": "", "quote_new": ""}
    ok = _j({**base, "old_value": e["old_value"] * 1.005, "new_value": e["new_value"]}, T3)
    assert ok["right"] and not ok["swapped"]
    bad = _j({**base, "old_value": e["old_value"] * 1.02, "new_value": e["new_value"]}, T3)
    assert not bad["right"]
    sw = _j({**base, "old_value": e["new_value"], "new_value": e["old_value"]}, T3)
    assert sw["swapped"] and not sw["right"]
    q0 = dict(T3, expect={**e, "old_value": 0})
    assert _j({**base, "old_value": 0, "new_value": e["new_value"]}, q0)["right"]
    assert not _j({**base, "old_value": 0.001, "new_value": e["new_value"]}, q0)["right"]
    assert not _j({**base, "old_value": "many", "new_value": e["new_value"]}, T3)["right"]


def test_stopped_or_unparsable_is_wrong_and_grounded_only_on_agent():
    rows = [
        {
            "model": "m",
            "condition": "agent",
            "qid": T1["id"],
            "content": "",
            "stopped": "max_turns",
            "labels": [],
        },
        {"model": "m", "condition": "agent", "qid": T1["id"], "content": "not json", "labels": []},
        {
            "model": "m",
            "condition": "agent",
            "qid": T1["id"],
            "content": json.dumps({"found": True, "old_article": T1["old_num"], "quote": ""}),
            "labels": [f"old:{T1['old_num']}"],
        },
        {
            "model": "m",
            "condition": "closed",
            "qid": T1["id"],
            "content": json.dumps({"found": True, "old_article": T1["old_num"], "quote": ""}),
            "labels": [],
        },
    ]
    s = score.score_rows(rows, Q)
    assert [r["right"] for r in s] == [False, False, True, True]
    assert [r["answer_grounded"] for r in s] == [False, False, True, None]
    assert [r["old_hit"] for r in s] == [False, False, True, None]
    assert s[0]["grade"] == "stopped"
    new = score.score_rows(
        [
            {
                "model": "m",
                "condition": "agent",
                "qid": T1_NEW["id"],
                "content": '{"found": false}',
                "labels": ["old:1"],
            }
        ],
        Q,
    )
    assert new[0]["old_hit"] is None and new[0]["right"]  # 新增: no old article to find


def test_primary_pairs_are_oracle_minus_agent_on_t2_t3():
    assert score.PRIMARY == [
        (m, "oracle", m, "agent", "T2+T3") for m in ("claude-haiku-5-5", "claude-opus-5-5")
    ]


def test_score_refuses_a_failed_stale_or_missing_cell(tmp_path, monkeypatch):
    monkeypatch.setattr(score, "RESULTS", tmp_path)
    qs = list(Q.values())
    good = [
        {
            "model": m,
            "condition": c,
            "qid": qid,
            "kind": Q[qid]["kind"],
            "chapter": Q[qid]["chapter"],
            "content": '{"found": false}',
            "error": "",
            "labels": [],
            "prompt_sha": run.signature(Q[qid], c),
        }
        for m, c, qid in run.grid(qs)
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
    diffs = (tmp_path / "diffs.csv").read_text("utf-8")
    n = sum(q["kind"] != "T1" for q in qs)
    assert f"oracle,claude-haiku-5-5,agent,T2+T3,True,right,{n}" in diffs  # T2 + T3 questions
