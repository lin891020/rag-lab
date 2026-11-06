"""The search tool is 17's cards retrieval. On the base corpus, asked with the question itself it
returns, byte for byte, the evidence 17 stored (and 26 answered from) for all 160 questions, with
the same hits. The big corpora add the five laws' units after the base ones, with scoped labels;
big-name puts 《法規名》 before every unit and every whole table, labels unchanged."""

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
    s = search.Searcher(_no_embed, search.EmbedCache(CACHE), corpus="base")
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


def test_big_adds_the_five_laws_after_the_base_units_with_scoped_labels():
    base_units, base_blocks = search.build_units("base")
    units, blocks = search.build_units("big")
    assert units[: len(base_units)] == base_units
    added = units[len(base_units) :]
    assert added and all(":" in u.labels[0] for u in added)
    assert {lab.split(":")[0].removeprefix("card") for u in added for lab in u.labels} <= {
        *search.laws.LAWS,
        "",
    }
    assert set(blocks) == set(base_blocks) | {k for k in blocks if ":" in k}
    assert "J0030019:表五○" in blocks and "J0030019:表一二九" in blocks and "表一二九" in blocks
    base_labels = {lab for u in base_units for lab in u.labels} | {
        lab for b in base_blocks.values() for lab in b.labels
    }
    added_labels = {lab for u in added for lab in u.labels} | {
        lab for k, b in blocks.items() if k not in base_blocks for lab in b.labels
    }
    assert not (base_labels & added_labels)


def test_big_name_prefixes_every_unit_and_table_with_its_law():
    big, big_blocks = search.build_units("big")
    named, named_blocks = search.build_units("big-name")
    assert [u.labels for u in named] == [u.labels for u in big]
    for b, n in zip(big, named, strict=True):
        law = search.law_of(b)
        assert n.text == f"《{law}》\n{b.text}"
    for k, b in big_blocks.items():
        assert named_blocks[k].text.startswith("《") and named_blocks[k].text.endswith(b.text)
        assert named_blocks[k].labels == b.labels
    assert search.law_of(big[0]) == "用戶用電設備裝置規則"


def test_unknown_corpus_is_refused():
    with pytest.raises(ValueError):
        search.build_units("huge")
