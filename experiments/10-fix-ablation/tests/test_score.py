from score import ABLATION, judge, numeral_kind, summarize


def _v(value, unit):
    return f'{{"quote": "x", "value": {value}, "unit": "{unit}"}}'


def test_gazette_units_have_synonyms():
    for want, got in [("個月", "月"), ("人", "名"), ("元", "新臺幣元"), ("次", "回"), ("日", "天")]:
        assert judge(_v(3, got), "value", {"value": 3, "unit": want})["value_ok"], (want, got)


def test_different_units_stay_different():
    for want, got in [("日", "月"), ("人", "次"), ("工作日", "日"), ("年", "個月"), ("元", "萬元")]:
        r = judge(_v(3, got), "value", {"value": 3, "unit": want})
        assert not r["value_ok"] and not r["value_lenient"], (want, got)


def test_lenient_forgives_only_an_empty_unit():
    r = judge(_v(5, ""), "value", {"value": 5, "unit": "條"})
    assert not r["value_ok"] and r["value_lenient"] and r["value_only_ok"]
    r = judge(_v(6, ""), "value", {"value": 5, "unit": "條"})
    assert not r["value_lenient"]
    r = judge('{"quote": "x", "articles": ["5"]}', "articles", {"articles": ["5"]})
    assert r["value_lenient"] == r["value_ok"]  # other types: same as strict


def test_numeral_kind_looks_at_how_the_gold_number_is_written():
    assert numeral_kind("每案補助新臺幣 5,000 元整", 5000) == "arabic"
    assert numeral_kind("銷售金額平均每人達新臺幣三萬六百元以上", 30600) == "wan"
    assert numeral_kind("儲槽頂部距離地面應在六十公分以上", 60) == "cn"
    assert numeral_kind("第五條 每案補助新臺幣 5,000 元整", 5000) == "arabic"  # 條號 ignored
    assert numeral_kind("四、汞：0.3ppm", 0.3) == "arabic"


def _row(model, cond, qid, ok, *, t="value", lenient=None):
    return {
        "model": model,
        "condition": cond,
        "qid": qid,
        "type": t,
        "numeral": "cn",
        "parse_ok": True,
        "schema_ok": True,
        "value_ok": ok,
        "value_only_ok": False,
        "value_lenient": ok if lenient is None else lenient,
        "evidence_has_gold": True,
        "eval_count": 1,
        "duration": 1.0,
    }


def test_summarize_reports_add_one_and_drop_one_diffs_on_both_scores():
    rows = []
    for qid in ("q1", "q2", "q3", "q4"):
        rows.append(_row("m", "fixed", qid, False, lenient=True))
        rows.append(_row("m", "+ocr", qid, True))
        rows.append(_row("m", "full", qid, True))
        rows.append(_row("m", "-ocr", qid, False))
    summary, diffs = summarize(rows)
    add = {(d["group"], d["a"], d["b"], d["metric"]): d for d in diffs}
    assert add[("add", "+ocr", "fixed", "value_ok")]["diff"] == 1.0
    assert add[("add", "+ocr", "fixed", "value_lenient")]["diff"] == 0.0
    assert add[("drop", "full", "-ocr", "value_ok")]["diff"] == 1.0
    assert ABLATION["add"][0] == "fixed" and ABLATION["drop"][0] == "full"
    kinds = {(r["condition"], r["type"]) for r in summary}
    assert ("fixed", "value:cn") in kinds  # numeral groups are reported
    assert any(r["metric"] == "value_lenient" for r in summary)
