from score import ABLATION, ANCHORS, CONDITIONS, judge, summarize


def _row(cond, qid, ok):
    return {
        "model": "m",
        "condition": cond,
        "qid": qid,
        "type": "value",
        "numeral": "cn",
        "parse_ok": True,
        "schema_ok": True,
        "value_ok": ok,
        "value_only_ok": False,
        "value_lenient": ok,
        "evidence_has_gold": True,
        "eval_count": 1,
        "duration": 1.0,
    }


def test_conditions_and_diffs_against_fixed_1000():
    assert CONDITIONS == ["closed-book", "oracle", "fixed-1000", "fixed-300", "struct"]
    assert ABLATION["chunking"] == ("fixed-1000", ["fixed-300", "struct"])
    assert ("oracle", "struct") in ANCHORS
    rows = []
    for qid in ("q1", "q2", "q3", "q4"):
        rows += [
            _row("fixed-1000", qid, False),
            _row("fixed-300", qid, True),
            _row("struct", qid, True),
        ]
    _, diffs = summarize(rows)
    d = {(x["group"], x["a"], x["b"], x["metric"]): x["diff"] for x in diffs}
    assert d[("chunking", "fixed-300", "fixed-1000", "value_ok")] == 1.0
    assert d[("chunking", "struct", "fixed-1000", "value_ok")] == 1.0


def test_points_are_judged_like_articles():
    exp = {"articles": ["4", "5", "10"]}
    assert judge('{"quote": "x", "articles": ["第四點", "5", "第十點"]}', "articles", exp)[
        "value_ok"
    ]
    assert not judge('{"quote": "x", "articles": ["4"]}', "articles", exp)["value_ok"]
