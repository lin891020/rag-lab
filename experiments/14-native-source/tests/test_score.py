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


def test_conditions_and_diffs_html_minus_pdf():
    assert CONDITIONS[2:] == ["pdf/fixed-300", "html/fixed-300", "pdf/points", "html+att/points"]
    assert ABLATION["fixed-300"] == ("pdf/fixed-300", ["html/fixed-300"])
    assert ("oracle", "html+att/points") in ANCHORS
    rows = []
    for qid in ("q1", "q2", "q3", "q4"):
        rows += [
            _row("pdf/fixed-300", qid, False),
            _row("html/fixed-300", qid, True),
            _row("pdf/points", qid, False),
            _row("html+att/points", qid, True),
        ]
    _, diffs = summarize(rows)
    d = {(x["group"], x["a"], x["b"], x["metric"]): x["diff"] for x in diffs}
    assert d[("fixed-300", "html/fixed-300", "pdf/fixed-300", "value_ok")] == 1.0
    assert d[("points", "html+att/points", "pdf/points", "value_ok")] == 1.0


def test_points_are_judged_like_articles():
    exp = {"articles": ["4", "5", "10"]}
    assert judge('{"quote": "x", "articles": ["第四點", "5", "第十點"]}', "articles", exp)[
        "value_ok"
    ]
    assert not judge('{"quote": "x", "articles": ["4"]}', "articles", exp)["value_ok"]
