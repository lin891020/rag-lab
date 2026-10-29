from score import abstained, grounded, score_rows, summarize


def test_p1_abstains_through_the_found_flag():
    assert abstained({"found": False, "quote": "", "value": 0, "unit": ""}, "value", "P1")
    assert not abstained({"found": True, "quote": "x", "value": 5, "unit": "日"}, "value", "P1")
    assert abstained({"found": False, "quote": "", "articles": []}, "articles", "P1")


def test_p0_abstention_is_inferred_from_empty_or_refusing_answers():
    assert abstained({"quote": "無明確規定", "value": 0, "unit": ""}, "value", "P0")
    assert abstained({"quote": "x", "value": 0, "unit": "元"}, "value", "P0")  # value 0
    assert abstained({"quote": "依據裡找不到相關條文", "value": 5, "unit": "日"}, "value", "P0")
    assert not abstained({"quote": "每案補助 5,000 元", "value": 5000, "unit": "元"}, "value", "P0")
    legit = "辦理工作能力評估結果為無工作能力者達百分之九十四以上"  # a lone 無 is not a refusal
    assert not abstained({"quote": legit, "value": 94, "unit": "%"}, "value", "P0")
    assert not abstained(
        {"quote": "未滿四十歲者，得每二年實施一次。", "value": 2, "unit": "年"}, "value", "P0"
    )
    assert abstained({"quote": "x", "articles": []}, "articles", "P0")
    assert abstained({"quote": "x", "law": ""}, "which", "P0")
    assert not abstained({"quote": "x", "law": "臺北市A辦法"}, "which", "P0")


def test_grounded_compares_without_whitespace_and_needs_a_real_quote():
    ev = "前項第三款所定評量表，應由醫院出具，並於\n出具後五日內送交保險人。"
    assert grounded("出具後五日內 送交保險人", ev)
    assert not grounded("出具後三日內送交保險人", ev)
    assert not grounded("", ev) and not grounded("出具", ev) and not grounded("出具後五日", None)


def _raw(
    prompt, evidence, qid, content, *, has_gold, leak=False, ev_text="每案補助新臺幣 5,000 元整"
):
    return {
        "model": "m",
        "prompt": prompt,
        "evidence": evidence,
        "qid": qid,
        "has_gold": has_gold,
        "leak": leak,
        "evidence_text": ev_text,
        "content": content,
        "eval_count": 1,
        "duration": 1.0,
    }


Q = {
    "q1": {
        "id": "q1",
        "type": "value",
        "quote": "每案補助新臺幣 5,000 元整",
        "expect": {"value": 5000, "unit": "元"},
    }
}
RIGHT = '{"quote": "每案補助新臺幣 5,000 元整", "value": 5000, "unit": "元"}'
MADE_UP = '{"quote": "每案補助新臺幣 8,000 元整", "value": 8000, "unit": "元"}'
LUCKY = '{"quote": "我記得是五千", "value": 5000, "unit": "元"}'


def test_score_rows_flags_answered_grounded_and_filtered_outcomes():
    rows = score_rows(
        [
            _raw("P0", "gold", "q1", RIGHT, has_gold=True),
            _raw("P0", "other-doc", "q1", MADE_UP, has_gold=False, ev_text="別份的字"),
            _raw("P0", "none", "q1", LUCKY, has_gold=False, ev_text=""),
        ],
        Q,
    )
    r_gold, r_other, r_none = rows
    assert r_gold["answered"] and r_gold["grounded"] and r_gold["value_ok"] and r_gold["value_ok_f"]
    assert r_other["answered"] and not r_other["grounded"] and not r_other["answered_f"]
    assert r_none["answered"] and r_none["value_ok"] and r_none["lucky"] and not r_none["grounded"]
    assert r_gold["group"] == "answerable" and r_other["group"] == "unanswerable"


def test_summarize_separates_the_two_groups_and_excludes_leaks():
    raw = []
    for qid in ("q1", "q2", "q3", "q4"):
        Q[qid] = Q["q1"] | {"id": qid}
        raw.append(_raw("P0", "same-doc", qid, MADE_UP, has_gold=False, ev_text="別處"))
        raw.append(
            _raw(
                "P1",
                "same-doc",
                qid,
                '{"found": false, "quote": "", "value": 0, "unit": ""}',
                has_gold=False,
                ev_text="別處",
            )
        )
        raw.append(_raw("P0", "gold", qid, RIGHT, has_gold=True))
        raw.append(
            _raw(
                "P1",
                "gold",
                qid,
                RIGHT.replace("5000", "5000").replace('"quote"', '"found": true, "quote"'),
                has_gold=True,
            )
        )
    raw.append(_raw("P0", "same-doc", "q5", MADE_UP, has_gold=False, leak=True, ev_text="別處"))
    Q["q5"] = Q["q1"] | {"id": "q5"}
    summary, diffs = summarize(score_rows(raw, Q))
    cell = {(r["prompt"], r["group"], r["evidence"], r["type"], r["metric"]): r for r in summary}
    assert cell[("P0", "unanswerable", "same-doc", "all", "answered")]["mean"] == 1.0
    assert (
        cell[("P0", "unanswerable", "same-doc", "all", "answered")]["n"] == 4
    )  # q5 (leak) excluded
    assert cell[("P1", "unanswerable", "same-doc", "all", "answered")]["mean"] == 0.0
    assert cell[("P1", "answerable", "gold", "all", "value_ok")]["mean"] == 1.0
    d = {(x["group"], x["metric"], x["a"], x["b"]): x for x in diffs}
    assert d[("unanswerable", "answered", "P1", "P0")]["diff"] == -1.0
    assert d[("answerable", "value_ok", "P1", "P0")]["diff"] == 0.0
    assert ("answerable", "value_ok_f", "filtered", "unfiltered") in d


def test_refusal_phrases_are_anchored_and_empty_payloads_count_as_abstention():
    legit = {"quote": "本規定不適用於未規定事項之情形", "value": 5, "unit": "日"}
    assert not abstained(
        legit, "value", "P0"
    )  # 不適用／未規定 inside statutory text is not a refusal
    assert abstained({"quote": "依據裡沒有相關規定", "value": 5, "unit": "日"}, "value", "P0")
    assert abstained({"found": True, "quote": "x", "law": ""}, "which", "P1")
    assert abstained({"found": True, "quote": "x", "articles": []}, "articles", "P1")
    assert abstained({"quote": "x", "law": "無"}, "which", "P0")
    assert abstained({"quote": "x", "law": "N/A"}, "which", "P0")


def test_score_rows_drops_unbuilt_windows_and_leak_checks_real_rows():
    raw = [
        _raw("P0", "same-doc", "q1", MADE_UP, has_gold=False)
        | {"done_reason": "not_constructible", "content": ""},
        _raw("P0", "real", "q1", MADE_UP, has_gold=False, ev_text="每案最高補助 5,000 元整"),
        _raw("P0", "real", "q1", MADE_UP, has_gold=False, ev_text="別處的字"),
    ]
    rows = score_rows(raw, Q)
    assert len(rows) == 2 and rows[0]["leak"] is True and rows[1]["leak"] is False
    summary, _ = summarize(rows)
    assert all(r["n"] == 1 for r in summary if r["group"] == "unanswerable")
    assert any(r["metric"] == "parse_ok" for r in summary)
