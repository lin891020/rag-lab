"""The search tool: 17's cards retrieval over one of three corpora -> (text, labels).

  base      27's index exactly: 用戶用電設備裝置規則, one chunk per article + one card per table
            (tests only; 27's rows are the baseline)
  big       base + the five laws of laws.py: their articles as units (labels <pcode>:<num>) and
            their tables as raw cards (keys <pcode>:<表號>), appended after the base units
  big-name  big with 《法規名》 on its own line before every unit and every whole table (10's fix)

Ranked by bge-m3 dense similarity to the query (top 50), cards replaced by their whole table,
packed to 3,000 characters skipping what does not fit (17's `budget_evidence(pack=True)`).
Stateless: the same query returns the same text. The embedding cache is this experiment's own
copy (seeded from 01's), so 16/17/27 keep theirs untouched.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import laws
from chunkers import Chunk, article_chunks, parse_articles
from evidence import DEPTH, budget_evidence, expand, load_corpus, table_cards, tables_view
from rag_lab import ollama
from retrievers import DenseIndex, EmbedCache, EmbedFn

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
EMB_CACHE = HERE.parents[1] / "data/28-hosted-bigger-corpus/emb-cache"  # seeded from 01's
EMBED_MODEL = "bge-m3"
BGE_DIGEST = "790764642607"  # 17's bge-m3; embedder() refuses another
BUDGET = 3000  # 17's cards
CORPORA = ("base", "big", "big-name")


def embedder() -> tuple[EmbedFn, EmbedCache]:
    digest = ollama.model_digest(EMBED_MODEL).split(":")[1][:12]
    if digest != BGE_DIGEST:
        raise SystemExit(f"bge-m3 is {digest}, 17 used {BGE_DIGEST}")
    return (lambda texts: ollama.embed(texts, model=EMBED_MODEL)), EmbedCache(
        EMB_CACHE / f"{digest}.npz"
    )


def law_of_label(key: str) -> str:
    """A label or table key -> the law's name: <pcode>:… is an added law, anything else base."""
    pcode, sep, _ = key.removeprefix("card:").partition(":")
    return laws.LAWS[pcode] if sep and pcode in laws.LAWS else laws.BASE_NAME


def law_of(unit: Chunk) -> str:
    return law_of_label(unit.labels[0])


def build_units(corpus: str) -> tuple[list[Chunk], dict[str, Chunk]]:
    if corpus not in CORPORA:
        raise ValueError(f"corpus {corpus!r}, expected one of {CORPORA}")
    law, raw = load_corpus()
    cards, blocks = table_cards(raw, tables_view(sub=True))
    units = article_chunks(parse_articles(law), max_len=1000) + cards
    if corpus == "base":
        return units, blocks
    added_cards: list[Chunk] = []
    for pcode, _name, arts, tables in laws.load_added():
        units += laws.law_units(pcode, arts)
        if tables:
            c, b = table_cards(tables, {})
            added_cards += c
            blocks.update(b)
    units += added_cards
    if corpus == "big-name":
        units = [replace(u, text=f"《{law_of(u)}》\n{u.text}") for u in units]
        blocks = {k: replace(b, text=f"《{law_of_label(k)}》\n{b.text}") for k, b in blocks.items()}
    return units, blocks


class Searcher:
    def __init__(self, embed: EmbedFn, cache: EmbedCache, *, corpus: str):
        self.corpus = corpus
        self.units, self.blocks = build_units(corpus)
        self.index = DenseIndex.build(self.units, embed, cache)
        self.cache = cache
        logger.info(
            "searcher",
            extra={"ctx": {"corpus": corpus, "units": len(self.units), "tables": len(self.blocks)}},
        )

    def search(self, query: str) -> tuple[str, set[str]]:
        ranked = expand([self.units[i] for i in self.index.search(query, DEPTH)], self.blocks)
        return budget_evidence(ranked, budget=BUDGET, pack=True)
