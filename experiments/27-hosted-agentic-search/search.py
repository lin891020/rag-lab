"""The search tool: 17's cards retrieval for any query -> (text, labels).

The same units as 17's cards cell (one chunk per article + one card per table), ranked by bge-m3
dense similarity to the query (top 50), cards replaced by their whole table, packed to 3,000
characters skipping what does not fit (17's `budget_evidence(pack=True)`). Stateless: the same
query returns the same text. Asked with the question itself it gives 17's cards evidence (test).
"""

from __future__ import annotations

import logging
from pathlib import Path

from chunkers import article_chunks, parse_articles
from evidence import DEPTH, budget_evidence, expand, load_corpus, table_cards, tables_view
from rag_lab import ollama
from retrievers import DenseIndex, EmbedCache, EmbedFn

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
EMB_CACHE = HERE.parents[1] / "data/01-chunking/emb-cache"  # 16's and 17's cache
EMBED_MODEL = "bge-m3"
BGE_DIGEST = "790764642607"  # 17's bge-m3; embedder() refuses another
BUDGET = 3000  # 17's cards


def embedder() -> tuple[EmbedFn, EmbedCache]:
    digest = ollama.model_digest(EMBED_MODEL).split(":")[1][:12]
    if digest != BGE_DIGEST:
        raise SystemExit(f"bge-m3 is {digest}, 17 used {BGE_DIGEST}")
    return (lambda texts: ollama.embed(texts, model=EMBED_MODEL)), EmbedCache(
        EMB_CACHE / f"{digest}.npz"
    )


class Searcher:
    def __init__(self, embed: EmbedFn, cache: EmbedCache):
        law, raw = load_corpus()
        cards, self.blocks = table_cards(raw, tables_view(sub=True))
        self.units = article_chunks(parse_articles(law), max_len=1000) + cards
        self.index = DenseIndex.build(self.units, embed, cache)
        self.cache = cache
        logger.info("searcher", extra={"ctx": {"units": len(self.units), "cards": len(cards)}})

    def search(self, query: str) -> tuple[str, set[str]]:
        ranked = expand([self.units[i] for i in self.index.search(query, DEPTH)], self.blocks)
        return budget_evidence(ranked, budget=BUDGET, pack=True)
