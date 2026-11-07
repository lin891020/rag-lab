"""The search tool: the old version (110-03-17) alone, one unit per article -> (text, labels).

Units are 28's `law_units` over `laws.parse_law(corpus/old-1100317.txt)`, prefix `old`
(labels old:16, old:13-1), bodies over 1,000 characters split at line boundaries with the header
repeated. Ranked by bge-m3 dense similarity to the query (top 50), packed to 3,000 characters
skipping what does not fit (17's `budget_evidence(pack=True)`, copied below). Stateless: the
same query returns the same text. The embedding cache is this experiment's own
(data/29-amendment-diff/emb-cache), seeded from nothing: the old version was never embedded.
The new version is not indexed: the question carries the new article's text.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
from pathlib import Path

import laws
from chunkers import Chunk
from rag_lab import ollama
from retrievers import DenseIndex, EmbedCache, EmbedFn

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
OLD_TXT = HERE / "corpus/old-1100317.txt"
EMB_CACHE = HERE.parents[1] / "data/29-amendment-diff/emb-cache"
EMBED_MODEL = "bge-m3"
BGE_DIGEST = "790764642607"  # 17's bge-m3; embedder() refuses another
BUDGET = 3000  # characters per search result (17's cards, 27's tool)
DEPTH = 50  # dense top-k before packing (27's)
PREFIX = "old"


def embedder() -> tuple[EmbedFn, EmbedCache]:
    digest = ollama.model_digest(EMBED_MODEL).split(":")[1][:12]
    if digest != BGE_DIGEST:
        raise SystemExit(f"bge-m3 is {digest}, 17 used {BGE_DIGEST}")
    return (lambda texts: ollama.embed(texts, model=EMBED_MODEL)), EmbedCache(
        EMB_CACHE / f"{digest}.npz"
    )


def old_articles() -> list[laws.LawArticle]:
    return laws.parse_law(OLD_TXT.read_text("utf-8"))


def build_units() -> list[Chunk]:
    return laws.law_units(PREFIX, old_articles(), max_len=1000)


@functools.cache
def index_sha() -> str:
    """sha256 of the built index (every unit's text and labels): a refetch or a parser fix makes
    earlier rows stale (28's)."""
    body = [[u.text, u.labels] for u in build_units()]
    return hashlib.sha256(json.dumps(body, ensure_ascii=False).encode("utf-8")).hexdigest()


def cut(c: Chunk, budget: int) -> Chunk:
    """17's: the first unit alone over budget is cut at a line boundary."""
    lines = c.text.splitlines()
    kept: list[str] = []
    used = 0
    for ln in lines:
        if used + len(ln) + 1 > budget:
            break
        kept.append(ln)
        used += len(ln) + 1
    return Chunk("\n".join(kept) if kept else c.text[:budget], c.labels, c.start, c.end, c.meta)


def budget_evidence(ranked: list[Chunk], *, budget: int, pack: bool = True) -> tuple[str, set[str]]:
    """17's: whole units in rank order while they fit; the first unit cut when it alone does not
    fit; pack=True skips a unit that does not fit and goes on. Returns (text, labels)."""
    kept: list[Chunk] = []
    used = 0
    for c in ranked:
        if used + len(c.text) > budget:
            if not kept:
                kept.append(cut(c, budget))
                used = len(kept[0].text)
                if not pack:
                    break
                continue
            if pack:
                continue
            break
        used += len(c.text)
        kept.append(c)
    labels = {lab for c in kept for lab in c.labels}
    text = "\n\n".join(f"【段 {i + 1}】\n{c.text}" for i, c in enumerate(kept))
    return text, labels


class Searcher:
    def __init__(self, embed: EmbedFn, cache: EmbedCache):
        self.units = build_units()
        self.index = DenseIndex.build(self.units, embed, cache)
        self.cache = cache
        logger.info("searcher", extra={"ctx": {"units": len(self.units), "prefix": PREFIX}})

    def search(self, query: str) -> tuple[str, set[str]]:
        ranked = [self.units[i] for i in self.index.search(query, DEPTH)]
        return budget_evidence(ranked, budget=BUDGET, pack=True)
