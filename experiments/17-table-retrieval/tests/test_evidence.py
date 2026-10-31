"""Evidence without any chat model: packing, cutting a whole table keeps only the rows that
survive, cards expand to their tables once, subtitles tell 表二五～二／三／四 apart, and the
struct@1500 cell reproduces 16's L1-struct hits question by question (from the embedding cache)."""

import json
from pathlib import Path

import pytest

from chunkers import Chunk
from evidence import (
    Evidence,
    budget_evidence,
    clean_table_block,
    cut,
    expand,
    hit_stats,
    interleave,
    load_corpus,
    rrf,
    subtitles,
    table_cards,
    tables_view,
)
from retrievers import EmbedCache
from rules import TABLES

HERE = Path(__file__).parents[1]
ROOT = HERE.parents[1]
Q = [json.loads(ln) for ln in (HERE / "questions.jsonl").read_text("utf-8").splitlines() if ln]


def _c(n: int, label: str) -> Chunk:
    return Chunk("x" * n, [label])


def test_stop_at_misfit_vs_pack():
    ranked = [_c(600, "a"), _c(600, "b"), _c(100, "c")]
    _, stop = budget_evidence(ranked, budget=1000)
    _, pack = budget_evidence(ranked, budget=1000, pack=True)
    assert stop == {"a"}
    assert pack == {"a", "c"}


def test_first_unit_is_cut_when_alone_too_long():
    text, labels = budget_evidence([_c(2000, "a"), _c(10, "b")], budget=1000)
    assert labels == {"a"} and text.count("x") == 1000


def test_cut_whole_table_keeps_only_surviving_rows():
    block = clean_table_block("表二五～五", tables_view(sub=True))
    small = cut(block, 400)
    assert len(small.text) <= 400
    assert "表二五～五|單線 1.6" in small.labels
    assert "表二五～五|絞線 22" not in small.labels  # far down the table
    assert "表二五～五|絞線 22" in block.labels


def test_budget_evidence_cuts_a_table_by_lines():
    block = clean_table_block("表二五～五", tables_view(sub=True))
    _, labels = budget_evidence([block], budget=400, pack=True)
    assert "表二五～五|單線 1.6" in labels and "表二五～五|絞線 22" not in labels


def test_subtitles_tell_the_metal_conduit_tables_apart():
    st = subtitles()
    assert "60℃" in st["表二五～二"] and "75℃" in st["表二五～三"] and "90℃" in st["表二五～四"]
    view = tables_view(sub=True)
    titles = {view[t]["title"] for t in ("表二五～二", "表二五～三", "表二五～四")}
    assert len(titles) == 3
    assert tables_view(sub=False) is TABLES  # 16's tables untouched


def test_cards_one_per_table_and_expand_once():
    _, raw = load_corpus()
    cards, blocks = table_cards(raw, tables_view(sub=True))
    assert len(cards) == len(blocks) == 109
    card = next(c for c in cards if c.labels == ["card:表二五～五"])
    assert "絞線 22" in card.text and "60℃" in card.text
    art = Chunk("第 25 條", ["25"])
    out = expand([card, art, card], blocks)
    assert [c.labels[0] for c in out] == [blocks["表二五～五"].labels[0], "25"]
    assert "表二五～五|絞線 22" in out[0].labels


def test_interleave_and_rrf():
    assert interleave([[1, 2, 3], [2, 4], [5]]) == [1, 2, 5, 4, 3]
    assert rrf([[1, 2], [2, 1]])[:2] in ([1, 2], [2, 1])
    assert rrf([[1, 2, 3], [2, 3, 1]])[0] == 2


def test_hit_stats_counts_tables_from_row_labels():
    q = {"gold": {"rows": [["表二五～五", "絞線 22"], ["表二五八～三", "20"]], "articles": ["214"]}}
    h = hit_stats({"表二五～五|絞線 8", "214"}, q)
    assert h == {
        "rows_hit": 0,
        "rows_total": 2,
        "tables_hit": 1,
        "tables_total": 2,
        "article_hit": True,
    }


def _no_embed(texts):
    raise AssertionError(f"cache miss: {len(texts)} texts")


@pytest.mark.skipif(
    not (ROOT / "data/01-chunking/emb-cache").exists(), reason="embedding cache lives on mini"
)
def test_struct_1500_reproduces_16_per_question():
    cache = EmbedCache(sorted((ROOT / "data/01-chunking/emb-cache").glob("*.npz"))[0])
    ev = Evidence(Q, cells=["struct@1500"], embed=_no_embed, cache=cache)
    raw16 = ROOT / "experiments/16-rule-lookup-compute/results/read/raw.jsonl"
    want = {
        r["qid"]: (r["rows_hit"], r["rows_total"])
        for r in map(json.loads, raw16.read_text("utf-8").splitlines())
        if r["condition"] == "L1-struct" and r["model"] == "qwen3:8b"
    }
    for q in Q:
        _, hit = ev.for_question("struct@1500", q)
        assert (hit["rows_hit"], hit["rows_total"]) == want[q["id"]], q["id"]
