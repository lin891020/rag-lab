from report import render_by_category, render_diffs, render_summary


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
            "a": "merged",
            "b": "page",
            "diff": 0.1,
            "lo": 0.02,
            "hi": 0.2,
            "separable": True,
        },
        {
            "model": "m",
            "metric": "value_ok",
            "a": "merged",
            "b": "pages",
            "diff": -0.03,
            "lo": -0.1,
            "hi": 0.05,
            "separable": False,
        },
    ]
    out = render_diffs(rows, model="m", metric="value_ok")
    assert "| merged − page | +.10 [+.02, +.20] | 分得出 |" in out
    assert "| merged − pages | −.03 [−.10, +.05] | 分不出 |" in out


def test_by_category_table():
    rows = [
        {
            "model": "m",
            "cat": "cont",
            "condition": "page",
            "mean": 0.8,
            "lo": 0.7,
            "hi": 0.9,
            "n": 45,
        },
        {
            "model": "m",
            "cat": "cont",
            "condition": "merged",
            "mean": 0.9,
            "lo": 0.8,
            "hi": 1.0,
            "n": 45,
        },
    ]
    out = render_by_category(rows, model="m")
    assert "| cont (45) |  | .80 [.70, .90] |  | .90 [.80, 1.0] |" in out
