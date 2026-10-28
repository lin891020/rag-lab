"""run.py bookkeeping and the three evidence shapes."""

import json
from pathlib import Path

from run import done_keys, drift_sample, evidence_for, grid, load_corpus

HERE = Path(__file__).parents[1]


def test_grid_is_models_x_conditions_x_questions():
    qs = [{"id": "a"}, {"id": "b"}]
    g = list(grid(["m1", "m2"], ["c1", "c2", "c3"], qs))
    assert len(g) == 12 and g[0] == ("m1", "c1", "a")


def test_done_keys_reads_existing_raw(tmp_path):
    p = tmp_path / "raw.jsonl"
    p.write_text(json.dumps({"model": "m", "condition": "c", "qid": "q", "content": "{}"}) + "\n")
    assert done_keys(p) == {("m", "c", "q")}
    assert done_keys(tmp_path / "missing.jsonl") == set()


def test_drift_sample_is_deterministic_and_small():
    qs = [{"id": f"q{i}"} for i in range(20)]
    s = drift_sample(["m1", "m2"], ["c1", "c2"], qs, per_cell=2)
    assert len(s) == 8 and s == drift_sample(["m1", "m2"], ["c1", "c2"], qs, per_cell=2)


def _q(qid):
    return next(
        json.loads(ln)
        for ln in (HERE / "questions.jsonl").read_text("utf-8").splitlines()
        if f'"{qid}"' in ln
    )


def test_evidence_shapes_for_a_continuation_question():
    corpus = load_corpus()
    q = _q("c18")  # 表五八七～三, page index 2
    page, label, info = evidence_for(q, "page", corpus)
    assert label == "表五八七～三 第 3 頁" and info["pages_used"] == [[2]]
    assert "表五八七～三 專供" not in page and "| 場所 |" not in page  # no title, no header
    pages, _, info2 = evidence_for(q, "pages", corpus)
    assert pages.count("【表五八七～三 第") == 3 and info2["pages_used"] == [[0, 1, 2]]
    merged, _, _ = evidence_for(q, "merged", corpus)
    assert merged.count("【") == 1 and "| 場所 |" in merged and "4.5m" in merged
    assert merged.count("\n| ") > pages.count("\n| ") - 3  # same rows, one header
