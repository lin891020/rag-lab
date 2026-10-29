from chunkers import Chunk
from run_rerank import CELLS, RERANK_TOP, RERANKERS, metrics_for, rerank_one
from run_read import budget_evidence


def _c(text, doc):
    return Chunk(text, [], 0, 0, {"doc": doc})


def test_metrics_for_reports_rank_based_and_budget_based_hits():
    gold = {"doc": "A", "quote": "並檢附下列文件"}
    ranked = [_c("甲" * 600, "B"), _c("並檢附下列文件" + "乙" * 300, "A"), _c("丙" * 100, "A")]
    m = metrics_for(ranked, gold)
    assert m["hit@1"] is False and m["hit@5"] is True and m["mrr"] == 0.5
    assert m["hit@1000"] is True and m["budget_has_gold"] == budget_evidence(ranked, gold)[1]
    none = metrics_for([_c("丙", "A")], gold)
    assert none["mrr"] == 0.0 and none["budget_has_gold"] is False


def test_budget_has_gold_uses_the_reading_tracks_rule_not_hit_at_budget():
    """A 1,200-char gold chunk ranked first: hit@1000 says no, the reader still gets its first
    1,000 chars (09's over_budget artifact), so budget_has_gold follows budget_evidence."""
    gold = {"doc": "A", "quote": "並檢附下列文件"}
    ranked = [_c("並檢附下列文件" + "乙" * 1200, "A")]
    m = metrics_for(ranked, gold)
    assert m["hit@1000"] is False and m["budget_has_gold"] is True


def test_grid_constants():
    assert list(CELLS) == ["full", "fixed"]
    assert CELLS["full"] == ("ocrrows", "article-cn+pre", True)
    assert RERANKERS == ["none", "bm25", "bge", "qwen3-rr", "listwise"]


class _Cell:
    def __init__(self, n):
        self.chunks = [_c(f"塊{i}", "A") for i in range(n)]

    def candidates(self, question):
        return list(range(len(self.chunks)))


class _Reverse:
    def rerank(self, question, chunks):
        return list(range(len(chunks)))[::-1], 0.5


def test_rerankers_only_reorder_the_top_window_and_keep_the_tail():
    ids, seconds = rerank_one(_Cell(50), "bge", _Reverse(), {"question": "q"})
    assert ids[:RERANK_TOP] == list(range(RERANK_TOP))[::-1]
    assert ids[RERANK_TOP:] == list(range(RERANK_TOP, 50)) and seconds == 0.5
    assert rerank_one(_Cell(50), "none", None, {"question": "q"})[0] == list(range(50))
