from report import render_diffs, render_summary

SUMMARY = [
    {"exp": "article", "chunking": "article", "retriever": "dense", "metric": "hit@5", "n": "30", "mean": "0.867", "lo": "0.733", "hi": "0.967"},
    {"exp": "article", "chunking": "article", "retriever": "dense", "metric": "hit@1000", "n": "30", "mean": "0.8", "lo": "0.667", "hi": "0.933"},
    {"exp": "article", "chunking": "fixed1000", "retriever": "bm25", "metric": "hit@5", "n": "30", "mean": "0.867", "lo": "0.733", "hi": "0.967"},
    {"exp": "article", "chunking": "fixed1000", "retriever": "bm25", "metric": "hit@1000", "n": "30", "mean": "0.633", "lo": "0.467", "hi": "0.8"},
    {"exp": "table", "chunking": "table-whole", "retriever": "dense", "metric": "hit@5", "n": "15", "mean": "0.933", "lo": "0.8", "hi": "1.0"},
    {"exp": "table", "chunking": "table-whole", "retriever": "dense", "metric": "hit@1000", "n": "15", "mean": "0.8", "lo": "0.6", "hi": "1.0"},
]
DIFFS = [
    {"exp": "article", "retriever": "dense", "metric": "hit@1000", "a": "article", "b": "fixed1000", "diff": "0.233", "lo": "0.067", "hi": "0.4", "separable": "True"},
    {"exp": "article", "retriever": "dense", "metric": "hit@1000", "a": "article", "b": "fixed269", "diff": "0.0", "lo": "-0.167", "hi": "0.133", "separable": "False"},
]


def test_render_summary_one_row_per_condition_with_intervals():
    md = render_summary(SUMMARY, exp="article", metrics=["hit@5", "hit@1000"])
    lines = md.splitlines()
    assert lines[0] == "| 切法 | 檢索 | hit@5 | hit@1000 |"
    assert "| article | dense | .87 [.73, .97] | .80 [.67, .93] |" in lines
    assert "| fixed1000 | bm25 | .87 [.73, .97] | .63 [.47, .80] |" in lines
    assert not any("table-whole" in ln for ln in lines)


def test_render_diffs_marks_separable_pairs():
    md = render_diffs(DIFFS, exp="article", retriever="dense", metric="hit@1000")
    assert "| article − fixed1000 | +.23 [+.07, +.40] | 分得出 |" in md
    assert "| article − fixed269 | +.00 [−.17, +.13] | 分不出 |" in md
