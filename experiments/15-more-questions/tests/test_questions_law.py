"""The new law bank for 01/07: ids after the reserve bank, gold inside chapter 2 or the 24 tables."""

import json
import re
from pathlib import Path

import pytest

HERE = Path(__file__).parents[1]
ONE = HERE.parent / "01-chunking"
BANK = HERE / "questions-law.jsonl"
pytestmark = pytest.mark.skipif(not BANK.exists(), reason="bank not written yet")


def _bank():
    return [json.loads(ln) for ln in BANK.read_text("utf-8").splitlines() if ln.strip()]


def test_ids_do_not_collide_and_are_in_order():
    old = set()
    for name in ("questions.jsonl", "questions-reserve.jsonl"):
        old |= {
            json.loads(ln)["id"]
            for ln in (ONE / name).read_text("utf-8").splitlines()
            if ln.strip()
        }
    qs = _bank()
    ids = [q["id"] for q in qs]
    assert not set(ids) & old and len(ids) == len(set(ids))
    assert [q["id"] for q in qs if q["type"] == "article"] == [f"a{n}" for n in range(41, 91)]
    assert [q["id"] for q in qs if q["type"] == "table"] == [f"t{n}" for n in range(18, 48)]


def test_gold_is_in_chapter_2_or_a_cited_table():
    tables = {p.stem for p in (ONE / "corpus/tables").glob("*.txt")}
    for q in _bank():
        assert q["why"].strip(), q["id"]
        assert isinstance(q["gold"], list) and q["gold"], q["id"]
        if q["type"] == "article":
            assert all(re.fullmatch(r"\d+", g) and 16 <= int(g) <= 110 for g in q["gold"]), q["id"]
        else:
            assert all(g in tables for g in q["gold"]), q["id"]


def test_article_golds_are_spread_and_tables_get_one_or_two():
    from collections import Counter

    qs = _bank()
    arts = Counter(q["gold"][0] for q in qs if q["type"] == "article")
    assert max(arts.values()) <= 2
    tabs = Counter(g for q in qs if q["type"] == "table" for g in q["gold"])
    assert max(tabs.values()) <= 2 and len(tabs) >= 20
