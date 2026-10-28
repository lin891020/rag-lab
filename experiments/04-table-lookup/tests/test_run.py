"""run.py bookkeeping: the grid, resume-from-raw, and the drift sample."""

import json

from run import done_keys, drift_sample, grid


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


def test_evidence_for_records_row_hit_and_cuts_rows():
    import json
    from pathlib import Path

    from run import evidence_for, load_tables

    here = Path(__file__).parents[1]
    tables = load_tables()
    q = next(
        json.loads(ln)
        for ln in (here / "questions.jsonl").read_text("utf-8").splitlines()
        if '"n54"' in ln
    )
    text, label, info = evidence_for(q, "md+range+rows", tables)
    assert label == "表九三～一" and info["row_hit"] is True and len(info["rows_kept"]) == 3
    assert "／下限 |" in text and "| 38 - 50 | 38 | 50 | 14 |" in text
    assert text.count("\n| ") == 4  # header + 3 rows
    whole, _, info2 = evidence_for(q, "md", tables)
    assert info2["row_hit"] is None and whole.count("\n| ") == 7
