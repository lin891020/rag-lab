"""Chunk building per cell, the quote-based hit, and the summary shape (no ollama)."""

from chunkers import Chunk
from retrievers import EmbedCache
from run_retrieval import (
    CHUNKINGS,
    build_cell,
    contains,
    first_hit_rank,
    hit_at_budget,
    norm,
    summarize,
)

GAZ = """農業部令 中華民國113年11月5日
修正「漁船船員管理規則」部分條文
第十四條之一　漁船船員手冊屆滿申請換發者，應填具申請書，並檢附下列文件。
一、最近一年內二吋半身正面脫帽照片四張。
第十五條　船員服務經許可者，其停留期間不得超過六個月。
"""
DOCS = [
    {"sysid": "A", "type": "final", "caption": "農業部令：修正「漁船船員管理規則」部分條文"},
    {"sysid": "B", "type": "final", "caption": "修正「嘉義縣特殊教育行政支持網絡聯繫及運作辦法」"},
]
TEXTS = {
    "A": GAZ,
    "B": "第一條　本辦法依特殊教育法第五十一條第三項規定訂定之。\n第二條　適用對象為學校。\n",
}


def test_norm_drops_whitespace_and_folds_width():
    assert norm("第 十四 條之一　漁船\n船員") == "第十四條之一漁船船員"
    assert norm("３,０００ 元") == "3,000元"


def test_contains_is_whitespace_insensitive():
    assert contains("應填具申請書，並檢附\n下列文件", "並檢附下列文件")
    assert not contains("應填具申請書", "並檢附下列文件")


def test_article_chunking_of_01_finds_nothing_in_a_gazette():
    chunks = build_cell(TEXTS, DOCS, chunking="article", prefix=False)
    assert chunks == []


def test_cn_chunking_tags_each_chunk_with_its_document():
    chunks = build_cell(TEXTS, DOCS, chunking="article-cn", prefix=False)
    assert [c.meta["doc"] for c in chunks] == ["A", "A", "B", "B"]
    assert chunks[0].labels == ["14-1"] and "漁船船員手冊屆滿" in chunks[0].text
    assert not any("農業部令" in c.text for c in chunks)  # the front matter is dropped


def test_cn_plus_pre_keeps_the_front_matter_as_its_own_chunk():
    chunks = build_cell(TEXTS, DOCS, chunking="article-cn+pre", prefix=False)
    pre = [c for c in chunks if c.labels == ["pre"]]
    assert len(pre) == 1 and pre[0].meta["doc"] == "A" and "部分條文" in pre[0].text


def test_prefix_puts_the_law_name_on_every_chunk():
    chunks = build_cell(TEXTS, DOCS, chunking="fixed-1000", prefix=True)
    assert all(c.text.startswith("《") for c in chunks)
    assert chunks[0].text.startswith("《漁船船員管理規則》") and chunks[-1].text.startswith(
        "《嘉義縣"
    )
    assert set(CHUNKINGS) == {"article", "article-cn", "article-cn+pre", "fixed-1000"}


def _c(text: str, doc: str) -> Chunk:
    return Chunk(text, [], 0, 0, {"doc": doc})


def test_hit_at_budget_walks_the_ranking_until_the_budget_is_spent():
    ranked = [_c("x" * 600, "B"), _c("並檢附下列文件" + "y" * 300, "A"), _c("並檢附下列文件", "A")]
    gold = {"doc": "A", "quote": "並檢附下列文件"}
    assert hit_at_budget(ranked, gold, budget=1000)  # 600 + 307 fits
    assert not hit_at_budget(
        ranked, gold, budget=800
    )  # second chunk does not fit, third never reached
    assert first_hit_rank(ranked, gold) == 1
    assert first_hit_rank(ranked, {"doc": "B", "quote": "並檢附下列文件"}) == -1


def test_summarize_pairs_every_cell_against_the_untouched_pipeline():
    raw = []
    for route, chunking, prefix, ret, hits in [
        ("base", "article", False, "dense", [0, 0, 1]),
        ("base", "fixed-1000", False, "dense", [1, 0, 1]),
        ("ocr", "article", False, "dense", [1, 1, 1]),
        ("base", "article", False, "random", [0, 0, 0]),
    ]:
        for i, h in enumerate(hits):
            raw.append(
                {
                    "route": route,
                    "chunking": chunking,
                    "prefix": prefix,
                    "retriever": ret,
                    "qid": f"q{i}",
                    "hit@5": float(h),
                    "hit@1000": float(h),
                }
            )
    summary, diffs = summarize(raw, metrics=["hit@1000"])
    assert len(summary) == 4
    base_diffs = [d for d in diffs if d["against"] == "base/article/noprefix"]
    assert {(d["route"], d["chunking"]) for d in base_diffs} == {
        ("base", "fixed-1000"),
        ("ocr", "article"),
    }
    assert all("random" != d["retriever"] for d in diffs)


def test_embed_cache_batches_by_characters_too():
    calls = []

    def embed(texts):
        calls.append(sum(len(t) for t in texts))
        return [[1.0, 0.0] for _ in texts]

    cache = EmbedCache(__import__("pathlib").Path("/nonexistent/x.npz"))
    texts = ["甲" * 9000, "乙" * 9000, "丙" * 9000, "丁" * 10]
    cache.fetch(texts, embed, batch=64, max_chars=20000)
    assert max(calls) <= 20000 and sum(calls) == 27010
