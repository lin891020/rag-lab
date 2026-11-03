"""posthoc counts reply formats per route, files identical between two corpora, per-table
answers and the questions that flip between two conditions, and what the opus pages cost."""

import json

from posthoc import by_table, flips, formats, opus_cost, same_files


def test_formats_count_pages_per_route():
    stats = {"r": {"t1": {"per_page": {"a": {"format": "html"}, "b": {"format": "plain"}}}}}
    assert formats(stats) == {"r": {"html": 1, "plain": 1}}


def test_same_files_compares_bytes(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    (a / "x.txt").write_text("1"), (b / "x.txt").write_text("1")
    (a / "y.txt").write_text("1"), (b / "y.txt").write_text("2")
    assert same_files(a, b) == (["x.txt"], ["y.txt"])


def _r(qid, cond, ok, model="m"):
    return {"model": model, "condition": cond, "qid": qid, "value_ok": ok}


def test_by_table_uses_the_gold_table():
    qs = [{"id": "q1", "gold": ["T1"]}, {"id": "q2", "gold": ["T1"]}, {"id": "q3", "gold": ["T2"]}]
    rows = [_r("q1", "c", True), _r("q2", "c", False), _r("q3", "c", True)]
    assert by_table(rows, qs, "m", "c") == {"T1": (1, 2), "T2": (1, 1)}


def test_flips_lists_both_directions():
    rows = [_r("q1", "a", True), _r("q1", "b", False), _r("q2", "a", False), _r("q2", "b", True)]
    rows += [_r("q3", "a", True), _r("q3", "b", True)]
    assert flips(rows, "m", "a", "b") == {"lost": ["q1"], "gained": ["q2"]}


def test_opus_cost_from_tokens(tmp_path):
    led = tmp_path / "l.jsonl"
    led.write_text(json.dumps({"tokens_in": 1_000_000, "tokens_out": 100_000}) + "\n")
    assert opus_cost(led) == (1_000_000, 100_000, 6.0)
