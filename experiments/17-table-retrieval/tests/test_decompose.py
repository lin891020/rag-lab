"""Sub-query parsing and the resumable file, without ollama."""

import json

from decompose import decompose, done_ids, new_numbers, parse


def test_parse_drops_empty_duplicates_and_caps():
    assert parse('{"queries": ["a", " a ", "", "b", "c", "d"]}') == ["a", "b", "c"]
    assert parse("not json") == []
    assert parse('{"queries": "a"}') == []


def test_new_numbers_ignores_thousands_separators():
    assert new_numbers("額定 3,500 A", ["3500A 接地導線"]) == []
    assert new_numbers("15 hp", ["15 hp 的 23.2 A"]) == ["23.2"]


def test_decompose_with_a_fake_chat():
    q = {"id": "c1-01", "chain": "C1", "question": "金屬管 22 mm²？"}
    row = decompose(q, chat=lambda *a, **k: {"content": '{"queries": ["金屬管安培容量表"]}'})
    assert row["queries"] == ["金屬管安培容量表"] and row["error"] == ""
    bad = decompose(q, chat=lambda *a, **k: {"content": "{}"})
    assert bad["queries"] == [] and bad["error"] == "unparsed"


def test_failed_rows_are_redone(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text(
        json.dumps({"qid": "a", "error": ""}) + "\n" + json.dumps({"qid": "b", "error": "x"}) + "\n"
    )
    assert done_ids(p) == {"a"}
