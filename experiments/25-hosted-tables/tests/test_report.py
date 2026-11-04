from report import render_diffs, render_summary


def test_summary_table_one_row_per_model_condition():
    rows = [
        {
            "model": "m",
            "condition": "schema",
            "metric": "parse_ok",
            "mean": 1.0,
            "lo": 1.0,
            "hi": 1.0,
        },
        {
            "model": "m",
            "condition": "schema",
            "metric": "value_ok",
            "mean": 0.8,
            "lo": 0.7,
            "hi": 0.9,
        },
    ]
    out = render_summary(rows, metrics=["parse_ok", "value_ok"])
    assert "| m | schema | 1.0 [1.0, 1.0] | .80 [.70, .90] |" in out


def _d(b, diff, lo, hi, loc, hic, primary=False):
    return {
        "model": "m",
        "metric": "value_ok",
        "a": "glm2-s50",
        "b": b,
        "diff": diff,
        "lo": lo,
        "hi": hi,
        "separable": str(lo > 0 or hi < 0),
        "lo_cluster": loc,
        "hi_cluster": hic,
        "separable_cluster": str(loc > 0 or hic < 0),
        "primary": str(primary),
    }


def test_diff_table_needs_both_intervals_and_marks_primary():
    rows = [
        _d("glm2-s100", 0.2, 0.1, 0.3, 0.05, 0.35, primary=True),
        _d("glm2-s83", 0.06, 0.01, 0.11, -0.02, 0.14),
        _d("glm2-s33", -0.03, -0.1, 0.05, -0.12, 0.07),
    ]
    out = render_diffs(rows, model="m", metric="value_ok")
    assert "| **glm2-s50 − glm2-s100** | +.20 [+.10, +.30] | [+.05, +.35] | 分得出 |" in out
    assert "| glm2-s50 − glm2-s83 | +.06 [+.01, +.11] | [−.02, +.14] | 分不出 |" in out
    assert "| glm2-s50 − glm2-s33 | −.03 [−.10, +.05] | [−.12, +.07] | 分不出 |" in out
    assert out.index("s100") < out.index("s33")  # primary pairs first
