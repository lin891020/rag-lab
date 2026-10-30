import json

from score import cluster_diff_ci, grade_ladder, grade_number, judge, score_rows, summarize

LADDER = [3.5, 5.5, 8.0, 14.0, 22.0]
Q_NUM = {
    "id": "q1",
    "chain": "C2",
    "cluster": "a",
    "expect": {"kind": "number", "value": 46.2, "unit": "A", "ladder": None},
}
Q_LAD = {
    "id": "q2",
    "chain": "C3",
    "cluster": "b",
    "expect": {"kind": "ladder", "value": 8.0, "unit": "mm²", "ladder": LADDER},
}
Q_ABS = {
    "id": "q3",
    "chain": "C3",
    "cluster": "c",
    "expect": {"kind": "abstain", "value": None, "unit": "mm²", "ladder": LADDER},
}


def _c(found=True, value=0, unit="", expression="", quote="x"):
    return json.dumps(
        {"found": found, "quote": quote, "expression": expression, "value": value, "unit": unit}
    )


def test_number_tolerance_and_units():
    assert grade_number(46.5, 46.2) == "match" and grade_number(47.0, 46.2) == "error"
    j = judge(_c(value=46.4, unit="安培"), Q_NUM)
    assert j["right"] and j["unit_ok"]
    j = judge(_c(value=46.4, unit=""), Q_NUM)
    assert not j["right"] and j["right_lenient"]
    j = judge(_c(value=46.4, unit="mm²"), Q_NUM)
    assert not j["right"] and not j["right_lenient"]


def test_ladder_grades_and_critical():
    assert grade_ladder(8.0, 8.0, LADDER) == "match"
    assert grade_ladder(14.0, 8.0, LADDER) == "acceptable_alternative"
    assert grade_ladder(22.0, 8.0, LADDER) == "error"
    assert grade_ladder(5.5, 8.0, LADDER) == "critical"
    assert grade_ladder(10.0, 8.0, LADDER) == "nonstandard"
    j = judge(_c(value=5.5, unit="mm²"), Q_LAD)
    assert j["critical"] and not j["right"]
    assert judge(_c(value=14, unit="mm²"), Q_LAD)["grade"] == "acceptable_alternative"


def test_abstain_questions_and_abstentions():
    assert (
        judge(_c(found=False), Q_ABS)["abstain_ok"]
        and not judge(_c(found=False), Q_ABS)["hard_answer"]
    )
    j = judge(_c(value=8, unit="mm²"), Q_ABS)
    assert j["hard_answer"] and not j["right"]
    j = judge(_c(found=False), Q_LAD)
    assert not j["answered"] and not j["right"] and not j["critical"]
    assert judge("not json", Q_NUM)["parse_ok"] is False


def test_summary_and_primary_diffs():
    qs = {"q1": Q_NUM, "q2": Q_LAD, "q3": Q_ABS}
    raw = []
    for cond, vals in (("L0", (0, 0, 8)), ("L1-struct", (46.2, 8, 0)), ("L2-struct", (46.2, 8, 0))):
        for q, v in zip(("q1", "q2", "q3"), vals, strict=True):
            raw.append(
                {
                    "model": "m",
                    "condition": cond,
                    "qid": q,
                    "chain": qs[q]["chain"],
                    "rows_hit": 1,
                    "rows_total": 1,
                    "expr_ok": True,
                    "content": _c(found=v != 0, value=v, unit=qs[q]["expect"]["unit"]),
                }
            )
    scored = score_rows(raw, qs)
    summary, diffs = summarize(scored)
    right = {(r["condition"], r["chain"]): r["mean"] for r in summary if r["metric"] == "right"}
    assert right[("L0", "all")] == 0.0 and right[("L1-struct", "all")] == 1.0
    hard = {
        r["condition"]: r["mean"]
        for r in summary
        if r["metric"] == "hard_answer" and r["chain"] == "all"
    }
    assert hard["L0"] == 1.0 and hard["L1-struct"] == 0.0
    prim = [
        d
        for d in diffs
        if d["primary"]
        and d["a"] == "L2-struct"
        and d["b"] == "L1-struct"
        and d["metric"] == "right"
    ]
    assert {d["scope"] for d in prim} == {"all", "C2+C3+C4"} and all(d["diff"] == 0.0 for d in prim)
    assert any(not d["primary"] and d["scope"] == "C3" for d in diffs)


def test_cluster_bootstrap_is_stratified_by_chain():
    diffs = {
        ("C2", "a"): [1.0] * 5,
        ("C2", "b"): [0.0] * 5,
        ("C3", "c"): [1.0] * 5,
        ("C3", "d"): [1.0] * 5,
    }
    d, lo, hi = cluster_diff_ci(diffs, n=300)
    assert abs(d - 0.75) < 1e-9 and lo <= d <= hi and hi - lo > 0
