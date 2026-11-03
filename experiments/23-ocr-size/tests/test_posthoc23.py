"""The fallback 'read at full size, re-read at s50 only when no table came back' is put together
from rows already asked: each question's evidence is its one gold table, so its answer under the
fallback is the s50 row when that table came back plain at s100, else the s100 row."""

from posthoc23 import fallback_rows, plain_tables


def test_plain_tables_are_those_with_any_plain_page():
    stats = {
        "glm2-s100": {
            "T1": {"per_page": {"T1-p0": {"format": "plain"}}},
            "T2": {"per_page": {"T2-p0": {"format": "html"}, "T2-p1": {"format": "plain"}}},
            "T3": {"per_page": {"T3-p0": {"format": "html"}}},
            "T4": {"per_page": {"T4-p0": {"format": "markdown"}}},  # still a table
        }
    }
    assert plain_tables(stats, "glm2-s100") == {"T1", "T2"}


def _r(qid, cond, ok):
    return {"model": "m", "condition": cond, "qid": qid, "value_ok": ok}


def test_fallback_takes_s50_only_for_plain_tables():
    qs = [{"id": "q1", "gold": ["T1"]}, {"id": "q2", "gold": ["T3"]}]
    rows = [_r("q1", "glm2-s100", False), _r("q1", "glm2-s50", True)]
    rows += [_r("q2", "glm2-s100", True), _r("q2", "glm2-s50", False)]
    got = fallback_rows(rows, qs, {"T1"})
    assert {(r["qid"], r["value_ok"], r["condition"]) for r in got} == {
        ("q1", True, "fallback"),
        ("q2", True, "fallback"),
    }
    assert {r["from"] for r in got} == {"glm2-s50", "glm2-s100"}
