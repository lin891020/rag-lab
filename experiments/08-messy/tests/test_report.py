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


def test_diff_table_marks_separable():
    rows = [
        {
            "model": "m",
            "metric": "value_ok",
            "a": "schema",
            "b": "prompt-only",
            "diff": 0.1,
            "lo": 0.02,
            "hi": 0.2,
            "separable": True,
        },
        {
            "model": "m",
            "metric": "value_ok",
            "a": "schema",
            "b": "json-mode",
            "diff": -0.03,
            "lo": -0.1,
            "hi": 0.05,
            "separable": False,
        },
    ]
    out = render_diffs(rows, model="m", metric="value_ok")
    assert "| schema − prompt-only | +.10 [+.02, +.20] | 分得出 |" in out
    assert "| schema − json-mode | −.03 [−.10, +.05] | 分不出 |" in out
