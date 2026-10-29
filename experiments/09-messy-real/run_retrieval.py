"""Retrieval track: thirty gazette documents in one index, 150 questions, every cell of
route x chunking x prefix x retriever.

    uv run python experiments/09-messy-real/run_retrieval.py   -> results/retrieval/

  route      base / ocr / rows            (extract.py; corpus/text/<sysid>.<route>.txt)
  chunking   article                      01's parser, untouched (expected: nothing parsed)
             article-cn                   headers written with Chinese numerals
             article-cn+pre               the same, plus the front matter (公告／總說明) kept as
                                          fixed-size chunks instead of dropped
             fixed-1000                   01's fallback, no parsing at all
  prefix     every chunk restated with 《law name》 first (07's title chain)
  retriever  dense (bge-m3) / bm25 (char bigrams) / random (noise floor)

A question hits when a ranked chunk comes from the question's document and contains its quote
(whitespace removed, widths folded). hit@5 counts the top five chunks; hit@1000 walks the ranking
while whole chunks still fit in 1,000 characters (01's budget).
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
import unicodedata
from collections.abc import Callable
from pathlib import Path

import numpy as np

from rag_lab import ollama
from rag_lab.log import setup_logging

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
CORPUS = HERE / "corpus"
RESULTS = HERE / "results/retrieval"
DATA = ROOT / "data/09-messy-real"
sys.path.insert(0, str(HERE))
from chunkers import (  # noqa: E402
    Chunk,
    article_chunks,
    fixed_chunks,
    parse_articles,
    parse_articles_cn,
    with_prefix,
)
from extract import law_name  # noqa: E402
from metrics import bootstrap_ci, paired_diff_ci  # noqa: E402
from retrievers import BM25Index, DenseIndex, EmbedCache, random_rankings  # noqa: E402

logger = logging.getLogger(__name__)

EMBED_MODEL = "bge-m3"
SEPS = ["\n\n", "\n", ". ", " ", ""]
ROUTES = ("base", "ocr", "rows")
CHUNKINGS = ("article", "article-cn", "article-cn+pre", "fixed-1000")
RETRIEVERS = ("dense", "bm25", "random")
K = 5
BUDGETS = (1000, 2000)
METRICS = ("hit@5", "hit@1000", "hit@2000")
DEPTH = 50
RANDOM_DRAWS = 20
_WS = re.compile(r"\s+")


def norm(s: str) -> str:
    return _WS.sub("", unicodedata.normalize("NFKC", s))


def contains(text: str, quote: str) -> bool:
    return norm(quote) in norm(text)


def is_gold(chunk: Chunk, gold: dict) -> bool:
    return chunk.meta.get("doc") == gold["doc"] and contains(chunk.text, gold["quote"])


def hit_at_k(ranked: list[Chunk], gold: dict, *, k: int) -> bool:
    return any(is_gold(c, gold) for c in ranked[:k])


def hit_at_budget(ranked: list[Chunk], gold: dict, *, budget: int) -> bool:
    used = 0
    for c in ranked:
        if used + len(c.text) > budget:
            break
        used += len(c.text)
        if is_gold(c, gold):
            return True
    return False


def first_hit_rank(ranked: list[Chunk], gold: dict) -> int:
    return next((i for i, c in enumerate(ranked) if is_gold(c, gold)), -1)


# -- chunks -------------------------------------------------------------------------------------


def doc_chunks(text: str, chunking: str) -> list[Chunk]:
    if chunking == "article":
        return article_chunks(parse_articles(text), max_len=1000)
    if chunking == "fixed-1000":
        return fixed_chunks(text, size=1000, overlap=0, separators=SEPS, spans=[])
    arts = parse_articles_cn(text)
    out = article_chunks(arts, max_len=1000)
    if chunking == "article-cn+pre":
        pre = text[: arts[0].start] if arts else text
        if pre.strip():
            front = fixed_chunks(pre, size=1000, overlap=0, separators=SEPS, spans=[])
            out = [Chunk(c.text, ["pre"], c.start, c.end, dict(c.meta)) for c in front] + out
    return out


def build_cell(
    texts: dict[str, str], docs: list[dict], *, chunking: str, prefix: bool
) -> list[Chunk]:
    """Every document's chunks for one (chunking, prefix), each tagged with its document."""
    out: list[Chunk] = []
    for d in docs:
        chunks = doc_chunks(texts[d["sysid"]], chunking)
        if prefix:
            chunks = with_prefix(chunks, law_name(d["caption"]))
        for c in chunks:
            c.meta["doc"] = d["sysid"]
        out.extend(chunks)
    return out


# -- evaluation ---------------------------------------------------------------------------------


def evaluate(
    chunks: list[Chunk], rank: Callable[[str, int], list[int]], questions: list[dict]
) -> list[dict]:
    rows = []
    for q in questions:
        ranked = [chunks[i] for i in rank(q["question"], DEPTH)]
        gold = {"doc": q["doc"], "quote": q["quote"]}
        row = {"qid": q["id"], "doc": q["doc"], "type": q["type"]}
        row["hit@5"] = float(hit_at_k(ranked, gold, k=K))
        for b in BUDGETS:
            row[f"hit@{b}"] = float(hit_at_budget(ranked, gold, budget=b))
        row["first_rank"] = first_hit_rank(ranked, gold)
        row["top5_docs"] = [c.meta.get("doc") for c in ranked[:K]]
        rows.append(row)
    return rows


def evaluate_random(chunks: list[Chunk], questions: list[dict], *, seed: int = 0) -> list[dict]:
    rankings = random_rankings(
        n_chunks=len(chunks), k=min(DEPTH, len(chunks)), draws=RANDOM_DRAWS, seed=seed
    )
    rows = []
    for q in questions:
        gold = {"doc": q["doc"], "quote": q["quote"]}
        acc = {"hit@5": 0.0, **{f"hit@{b}": 0.0 for b in BUDGETS}}
        for r in rankings:
            ranked = [chunks[i] for i in r]
            acc["hit@5"] += hit_at_k(ranked, gold, k=K)
            for b in BUDGETS:
                acc[f"hit@{b}"] += hit_at_budget(ranked, gold, budget=b)
        rows.append(
            {
                "qid": q["id"],
                "doc": q["doc"],
                "type": q["type"],
                **{m: v / RANDOM_DRAWS for m, v in acc.items()},
                "first_rank": -1,
                "top5_docs": [],
            }
        )
    return rows


def gold_counts(chunks: list[Chunk], questions: list[dict]) -> dict[str, dict]:
    """Per question: chunks of its document, and how many of them hold the quote."""
    out = {}
    for q in questions:
        gold = {"doc": q["doc"], "quote": q["quote"]}
        mine = [c for c in chunks if c.meta.get("doc") == q["doc"]]
        out[q["id"]] = {
            "doc_chunks": len(mine),
            "gold_chunks": sum(is_gold(c, gold) for c in mine),
        }
    return out


def drift_probe(cache: EmbedCache, chunks: list[Chunk], *, n: int = 20) -> float:
    sample = [c.text for c in chunks[:: max(1, len(chunks) // n)]][:n]
    fresh = np.asarray(ollama.embed(sample, model=EMBED_MODEL), dtype=np.float32)
    cached = np.stack([cache.vecs[EmbedCache.key(t)] for t in sample])
    cos = (fresh * cached).sum(1) / (np.linalg.norm(fresh, axis=1) * np.linalg.norm(cached, axis=1))
    return float(cos.min())


def _cell(r: dict) -> tuple:
    return (r["route"], r["chunking"], bool(r["prefix"]), r["retriever"])


def summarize(raw: list[dict], *, metrics: list[str]) -> tuple[list[dict], list[dict]]:
    """Summary per cell; paired diffs of every cell against the untouched pipeline (same
    retriever) and, for each single change, against the cell without it."""
    by: dict[tuple, list[dict]] = {}
    for r in raw:
        by.setdefault(_cell(r), []).append(r)
    summary = []
    for cell, rows in sorted(by.items()):
        route, ch, pre, ret = cell
        for m in metrics:
            mean, lo, hi = bootstrap_ci([r[m] for r in rows])
            summary.append(
                {
                    "route": route,
                    "chunking": ch,
                    "prefix": pre,
                    "retriever": ret,
                    "metric": m,
                    "n": len(rows),
                    "mean": round(mean, 3),
                    "lo": round(lo, 3),
                    "hi": round(hi, 3),
                }
            )

    def diff(cell: tuple, other: tuple, against: str) -> list[dict]:
        if other not in by or other == cell:
            return []
        out = []
        for m in metrics:
            a = [r[m] for r in sorted(by[cell], key=lambda r: r["qid"])]
            b = [r[m] for r in sorted(by[other], key=lambda r: r["qid"])]
            d, lo, hi = paired_diff_ci(a, b)
            out.append(
                {
                    "route": cell[0],
                    "chunking": cell[1],
                    "prefix": cell[2],
                    "retriever": cell[3],
                    "against": against,
                    "metric": m,
                    "diff": round(d, 3),
                    "lo": round(lo, 3),
                    "hi": round(hi, 3),
                    "separable": lo > 0 or hi < 0,
                }
            )
        return out

    diffs = []
    for cell in sorted(by):
        route, ch, pre, ret = cell
        if ret == "random":
            continue
        diffs += diff(cell, ("base", "article", False, ret), "base/article/noprefix")
        if pre:
            diffs += diff(cell, (route, ch, False, ret), "same/noprefix")
        if route != "base":
            diffs += diff(cell, ("base", ch, pre, ret), "base/same")
        if ch != "article":
            diffs += diff(cell, (route, "article", pre, ret), "same/article")
    return summary, diffs


# -- main -----------------------------------------------------------------------------------------


def load_questions() -> list[dict]:
    return [
        json.loads(ln)
        for ln in (HERE / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]


def main() -> int:
    setup_logging("rag_lab")
    RESULTS.mkdir(parents=True, exist_ok=True)
    docs = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))["docs"]
    questions = load_questions()
    digest = ollama.model_digest(EMBED_MODEL)
    cache = EmbedCache(DATA / "emb-cache" / f"{digest.split(':')[1][:12]}.npz")
    embed = lambda texts: ollama.embed(texts, model=EMBED_MODEL)  # noqa: E731
    raw: list[dict] = []
    stats: dict = {}
    first_chunks: list[Chunk] | None = None
    for route in ROUTES:
        texts = {
            d["sysid"]: (CORPUS / "text" / f"{d['sysid']}.{route}.txt").read_text(encoding="utf-8")
            for d in docs
        }
        for chunking in CHUNKINGS:
            for prefix in (False, True):
                chunks = build_cell(texts, docs, chunking=chunking, prefix=prefix)
                key = f"{route}/{chunking}/{'prefix' if prefix else 'noprefix'}"
                counts = gold_counts(chunks, questions)
                stats[key] = {
                    "chunks": len(chunks),
                    "docs_with_chunks": len({c.meta["doc"] for c in chunks}),
                    "questions_with_gold_chunk": sum(v["gold_chunks"] > 0 for v in counts.values()),
                    "gold": counts,
                }
                if not chunks:
                    for ret in RETRIEVERS:
                        for q in questions:
                            raw.append(
                                {
                                    "route": route,
                                    "chunking": chunking,
                                    "prefix": prefix,
                                    "retriever": ret,
                                    "qid": q["id"],
                                    "doc": q["doc"],
                                    "type": q["type"],
                                    "hit@5": 0.0,
                                    **{f"hit@{b}": 0.0 for b in BUDGETS},
                                    "first_rank": -1,
                                    "top5_docs": [],
                                }
                            )
                    logger.warning("empty cell", extra={"ctx": {"cell": key}})
                    continue
                if first_chunks is None:
                    first_chunks = chunks
                dense = DenseIndex.build(chunks, embed, cache)
                cache.save()
                bm25 = BM25Index(chunks)
                for ret, rows in (
                    ("dense", evaluate(chunks, dense.search, questions)),
                    ("bm25", evaluate(chunks, bm25.search, questions)),
                    ("random", evaluate_random(chunks, questions)),
                ):
                    for r in rows:
                        raw.append(
                            {
                                "route": route,
                                "chunking": chunking,
                                "prefix": prefix,
                                "retriever": ret,
                                **r,
                            }
                        )
                logger.info("cell done", extra={"ctx": {"cell": key, "chunks": len(chunks)}})
    with (RESULTS / "raw.jsonl").open("w", encoding="utf-8") as f:
        for r in raw:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary, diffs = summarize(raw, metrics=list(METRICS))
    for name, rows in (("summary.csv", summary), ("diffs.csv", diffs)):
        with (RESULTS / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    drift = drift_probe(cache, first_chunks) if first_chunks else None
    meta = {
        "embed_model": EMBED_MODEL,
        "embed_digest": digest,
        "ollama": ollama.version(),
        "n_questions": len(questions),
        "n_docs": len(docs),
        "depth": DEPTH,
        "k": K,
        "budgets": list(BUDGETS),
        "random_draws": RANDOM_DRAWS,
        "drift_min_cosine": drift,
        "cells": {k: {kk: vv for kk, vv in v.items() if kk != "gold"} for k, v in stats.items()},
    }
    (RESULTS / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (RESULTS / "gold_counts.json").write_text(
        json.dumps({k: v["gold"] for k, v in stats.items()}, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("retrieval done", extra={"ctx": {"rows": len(raw), "drift": drift}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
