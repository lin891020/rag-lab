"""Evidence without any chat model: 17's packing, cutting, cards and subtitles, plus 18's kv
rewrite: short column names, a kv table carries exactly the rows its markdown twin carries (whole
or cut), and cards-kv holds the same rows as cards question by question (from the embedding
cache)."""

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
    kv_like,
    short_names,
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


def test_short_names_keep_only_what_differs_in_a_group():
    assert short_names(TABLES["表二五～二"]["names"]) == [
        "線別",
        "標稱截面積（mm2)",
        "根數/直徑（mm）",
        "3以下",
        "4",
        "5-6",
        "7-9",
    ]


def test_short_names_prefix_every_clash_and_never_repeat():
    sn = short_names(TABLES["表二五八～三"]["names"])
    assert sn[0] == "馬力(hp)" and sn[1] == "115"
    assert sn[5] == "鼠籠型及繞線型感應電動機註1(4極)／380"
    assert sn[8] == "功率因數為1註2之同步型電動機／380"
    for tid, t in TABLES.items():  # 表四○'s source header repeats one name four times
        if len(set(t["names"])) == len(t["names"]):
            assert len(set(short_names(t["names"]))) == len(t["names"]), tid


def test_kv_like_keeps_the_markdown_rows_whole_or_cut():
    view = tables_view(sub=True)
    for tid in view:
        md = clean_table_block(tid, view)
        assert set(kv_like(md, view).labels) == set(md.labels), tid
    md = clean_table_block("表二五～二", view)
    kv = kv_like(md, view)
    assert "絞線 3.5" not in kv.text  # rows are kv lines, not pipes
    assert "線別=絞線；標稱截面積（mm2)=3.5；根數/直徑（mm）=7/0.8；3以下=19；4=17" in kv.text
    assert kv.text.splitlines()[0] == md.text.splitlines()[0]  # title (with subtitle) kept
    small = cut(md, 400)
    kv_small = kv_like(small, view)
    assert set(kv_small.labels) == set(small.labels)
    assert len(kv_small.text.splitlines()) == len(small.text.splitlines())


def test_kv_like_passes_articles_and_raw_tables_through():
    art = Chunk("第 25 條 …", ["25"])
    assert kv_like(art, tables_view(sub=True)) is art


@pytest.mark.skipif(
    not (ROOT / "data/01-chunking/emb-cache").exists(), reason="embedding cache lives on mini"
)
def test_cards_kv_holds_the_rows_cards_holds():
    cache = EmbedCache(sorted((ROOT / "data/01-chunking/emb-cache").glob("*.npz"))[0])
    ev = Evidence(Q, cells=["cards", "cards-kv", "cards-kv@3000"], embed=_no_embed, cache=cache)
    longer = 0
    for q in Q:
        md, hit_md = ev.for_question("cards", q)
        kv, hit_kv = ev.for_question("cards-kv", q)
        assert hit_kv == hit_md, q["id"]
        assert md.count("【段") == kv.count("【段"), q["id"]
        longer += len(kv) > len(md)
        _, hit3 = ev.for_question("cards-kv@3000", q)
        assert hit3["rows_hit"] <= hit_md["rows_total"]
    assert longer > 0
