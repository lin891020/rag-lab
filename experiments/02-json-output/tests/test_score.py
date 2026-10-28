"""Scoring is pure string/number work; every rule the README relies on is pinned here."""

from score import judge, normalize_unit

EXPECT = {"value": 3.5, "unit": "mm²", "alt": []}
PCT = {
    "value": 75,
    "unit": "%",
    "alt": [{"value": 0.75, "unit": ""}, {"value": 0.75, "unit": "倍"}],
}


def test_normalize_unit_folds_spelling_variants():
    assert normalize_unit("mm^2") == normalize_unit("mm²") == normalize_unit("平方毫米") == "mm²"
    assert normalize_unit(" A ") == normalize_unit("安培") == "A"
    assert normalize_unit("％") == "%"
    assert normalize_unit("") == ""
    assert normalize_unit("furlong") == "furlong"  # unknown units pass through


def test_valid_object_scores_everything_true():
    r = judge('{"quote": "絞線截面積不得小於三‧五平方毫米", "value": 3.5, "unit": "mm2"}', EXPECT)
    assert r == {
        "parse_ok": True,
        "schema_ok": True,
        "value_ok": True,
        "value_only_ok": False,
        "fenced": False,
    }


def test_fenced_json_parses_but_is_flagged():
    r = judge('```json\n{"quote": "x", "value": 3.5, "unit": "mm²"}\n```', EXPECT)
    assert r["parse_ok"] and r["value_ok"] and r["fenced"]


def test_empty_or_prose_fails_parse_only():
    for bad in ("", "答案是 3.5 平方毫米", "{value: 3.5}"):
        r = judge(bad, EXPECT)
        assert r == {
            "parse_ok": False,
            "schema_ok": False,
            "value_ok": False,
            "value_only_ok": False,
            "fenced": False,
        }, bad


def test_missing_key_or_string_value_breaks_schema_not_value():
    r = judge('{"value": "3.5", "unit": "mm²"}', EXPECT)
    assert r["parse_ok"] and not r["schema_ok"] and r["value_ok"]


def test_wrong_unit_counts_as_value_only():
    r = judge('{"quote": "", "value": 3.5, "unit": "mm"}', EXPECT)
    assert r["schema_ok"] and not r["value_ok"] and r["value_only_ok"]


def test_wrong_value_fails_both():
    r = judge('{"quote": "", "value": 4, "unit": "mm²"}', EXPECT)
    assert r["schema_ok"] and not r["value_ok"] and not r["value_only_ok"]


def test_alt_pairs_are_accepted():
    assert judge('{"quote": "", "value": 0.75, "unit": ""}', PCT)["value_ok"]
    assert judge('{"quote": "", "value": 75, "unit": "%"}', PCT)["value_ok"]
    assert not judge('{"quote": "", "value": 0.75, "unit": "%"}', PCT)["value_ok"]


def test_tolerance_is_tiny():
    assert judge('{"quote": "", "value": 3.5000000001, "unit": "mm²"}', EXPECT)["value_ok"]
    assert not judge('{"quote": "", "value": 3.51, "unit": "mm²"}', EXPECT)["value_ok"]


def test_non_object_json_is_parse_ok_but_nothing_else():
    r = judge("[3.5]", EXPECT)
    assert r["parse_ok"] and not r["schema_ok"] and not r["value_ok"]
