"""Scoring is pure string/number work; every rule the README relies on is pinned here."""

import pytest


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
    assert normalize_unit("V·A") == "VA"  # qwen3 wrote the SI dot form
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


def test_pairs_put_the_fixed_rendering_first():
    """diffs are a − b with clean, then glm2-flat, first: + = what flattening gained, or what the
    OCR route still costs against the text layer."""
    from score import PAIRS, PRIMARY

    assert PRIMARY == [("glm2-flat", "glm2")]
    assert PAIRS[:1] == PRIMARY and len(set(PAIRS)) == len(PAIRS)
    rank = {"clean": 0, "glm2-flat": 1, "glm2": 2}
    for a, b in PAIRS:
        assert rank[a] < rank[b], (a, b)


def test_cluster_ci_is_wider_when_questions_of_a_table_move_together():
    from score import cluster_diff_ci, paired_diff_ci

    # 10 tables x 5 questions; each table either all lost (1) or all kept (0)
    by_table = {f"t{i}": [1.0 if i < 3 else 0.0] * 5 for i in range(10)}
    flat = [v for vs in by_table.values() for v in vs]
    d, lo, hi = cluster_diff_ci(by_table)
    _, plo, phi = paired_diff_ci(flat, [0.0] * len(flat))
    assert d == pytest.approx(0.3) and lo <= plo and hi >= phi and hi - lo > phi - plo


def test_summarize_marks_primary_diffs_with_both_intervals():
    from score import summarize

    qs = {f"q{i}": {"gold": [f"t{i % 4}"]} for i in range(8)}
    rows = []
    for cond, ok in (("glm2-flat", 1), ("glm2", 0)):
        for q in qs:
            rows.append(
                {
                    "model": "m",
                    "condition": cond,
                    "qid": q,
                    "parse_ok": True,
                    "schema_ok": True,
                    "value_ok": bool(ok),
                    "value_only_ok": False,
                }
            )
    _, diffs = summarize(rows, qs)
    d = next(r for r in diffs if (r["a"], r["b"], r["metric"]) == ("glm2-flat", "glm2", "value_ok"))
    assert d["primary"] and d["diff"] == 1.0 and d["clusters"] == 4
    assert d["separable"] and d["separable_cluster"]


def test_score_refuses_while_a_call_failed_or_a_prompt_is_stale(tmp_path, monkeypatch):
    import json

    import run
    import score

    q = run.load_questions()[0]
    lens = run.prompt_lens([q], run.load_corpus())
    good = {"model": "m", "condition": "clean", "qid": q["id"], "content": "{}", "error": ""}
    monkeypatch.setattr(score, "RESULTS", tmp_path)
    for row in (
        dict(good, prompt_len=lens[("clean", q["id"])], error="TimeoutError()"),
        dict(good, prompt_len=lens[("clean", q["id"])] + 1),
    ):
        (tmp_path / "raw.jsonl").write_text(json.dumps(row) + "\n")
        assert score.main() == 1
    ok = [dict(good, condition=c, prompt_len=lens[(c, q["id"])]) for c in ("glm2", "glm2-flat")]
    (tmp_path / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ok))
    assert score.main() == 0
