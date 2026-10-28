"""run.py bookkeeping: the grid, resume-from-raw, and the drift sample."""

import json

from run_table import done_keys, drift_sample, grid


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
