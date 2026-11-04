"""The search tool is 17's cards retrieval: asked with the question itself it returns, byte for
byte, the evidence 17 stored (and 26 answered from) for all 160 questions, with the same hits."""

import gzip
import json
from pathlib import Path

import pytest

import search
from evidence import hit_stats

RAW17 = Path(__file__).parents[2] / "17-table-retrieval/results/read/raw.jsonl.gz"
CACHE = search.EMB_CACHE / f"{search.BGE_DIGEST}.npz"


def _no_embed(texts):
    raise AssertionError(f"not in the cache: {texts[:1]}")


@pytest.mark.skipif(not CACHE.exists(), reason="the embedding cache lives on the mini only")
def test_the_question_as_query_gives_17s_cards_evidence():
    stored = {}
    with gzip.open(RAW17, "rt", encoding="utf-8") as f:
        for ln in f:
            r = json.loads(ln)
            if r["model"] == "gpt-oss:20b" and r["condition"] == "cards" and not r.get("error"):
                stored[r["qid"]] = r
    s = search.Searcher(_no_embed, search.EmbedCache(CACHE))
    qs = [json.loads(ln) for ln in (search.HERE / "questions.jsonl").open(encoding="utf-8")]
    assert len(qs) == len(stored) == 160
    for q in qs:
        text, labels = s.search(q["question"])
        assert text == stored[q["id"]]["evidence_text"], q["id"]
        h = hit_stats(labels, q)
        assert (h["rows_hit"], h["rows_total"]) == (
            stored[q["id"]]["rows_hit"],
            stored[q["id"]]["rows_total"],
        ), q["id"]


def test_search_is_stateless_and_packs_to_the_budget():
    from chunkers import Chunk

    class _Index:
        def search(self, query, k):
            return [0, 1, 2]

    s = search.Searcher.__new__(search.Searcher)
    s.units = [
        Chunk("a" * 2000, ["1"]),
        Chunk("b" * 2000, ["2"]),  # does not fit after the first: skipped (cards pack)
        Chunk("c" * 500, ["3"]),
    ]
    s.blocks, s.index = {}, _Index()
    one, two = s.search("x"), s.search("x")
    assert one == two and one[1] == {"1", "3"} and len(one[0]) <= search.BUDGET + 40
