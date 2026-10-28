"""01's retrieval harness with the chunking survey's cheap recommendations as conditions.

    uv run python experiments/07-chunking-2/run.py

Article questions only (01's 30 + the 10 reserve article questions). Tables are always
table-whole (01's winner for tables). Text conditions:
  article          01's best: one chunk per 條 with the '章 / 節 / 第 N 條' header   (baseline)
  article-noprefix the same chunks without that header line                        (title chain)
  article+context  article + one LLM-written sentence (context.py) in front        (contextual)
  page             one chunk per PDF page                                           (NVIDIA)
  small-to-big     index <=269-char pieces cut inside each 條 / table, return the whole parent
Retrievers: dense (bge-m3), bm25 (char bigrams), hybrid (RRF of both), random (noise floor).
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import platform
import statistics
import sys
from collections.abc import Callable
from datetime import date
from importlib.metadata import version as pkg_version
from pathlib import Path

import numpy as np
from pypdf import PdfReader

from rag_lab import ollama
from rag_lab.log import setup_logging
from rag_lab.pdf import extract_text

sys.path.insert(0, str(Path(__file__).parent))
from chunkers import (  # noqa: E402
    Chunk,
    article_chunks,
    page_chunks,
    parse_articles,
    small_to_big,
    strip_prefix,
    table_whole_chunks,
    with_context,
)
from context import load_contexts  # noqa: E402
from metrics import bootstrap_ci, hit_at_budget, hit_at_k, paired_diff_ci  # noqa: E402
from retrievers import (  # noqa: E402
    BM25Index,
    DenseIndex,
    EmbedCache,
    HybridIndex,
    parent_rank,
    random_rankings,
)

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
ONE = HERE.parent / "01-chunking"
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
EMBED_MODEL = "bge-m3"
SEPS = ["\n\n", "\n", ". ", " ", ""]
SMALL = 269  # 01's size-matched fixed chunk
K = 5
BUDGETS = [1000, 2000]
RANDOM_DRAWS = 200
RETRIEVERS = ["dense", "bm25", "hybrid", "random"]


def evaluate(
    chunks: list[Chunk],
    rank: Callable[[str, int], list[int]],
    questions: list[dict],
    *,
    k: int,
    budgets: list[int],
) -> list[dict]:
    rows = []
    depth = max(k, 50)
    for q in questions:
        ranked = [chunks[i] for i in rank(q["question"], depth)]
        row = {"qid": q["id"], f"hit@{k}": float(hit_at_k(ranked, q["gold"], k=k))}
        for b in budgets:
            row[f"hit@{b}"] = float(hit_at_budget(ranked, q["gold"], budget=b))
        row["top"] = [c.labels for c in ranked[:k]]
        rows.append(row)
    return rows


def evaluate_random(
    chunks: list[Chunk], questions: list[dict], *, k: int, budgets: list[int], draws: int, seed: int
) -> list[dict]:
    depth = max(k, 50)
    rankings = random_rankings(
        n_chunks=len(chunks), k=min(depth, len(chunks)), draws=draws, seed=seed
    )
    rows = []
    for q in questions:
        acc = {f"hit@{k}": 0.0, **{f"hit@{b}": 0.0 for b in budgets}}
        for r in rankings:
            ranked = [chunks[i] for i in r]
            acc[f"hit@{k}"] += hit_at_k(ranked, q["gold"], k=k)
            for b in budgets:
                acc[f"hit@{b}"] += hit_at_budget(ranked, q["gold"], budget=b)
        rows.append({"qid": q["id"], **{m: v / draws for m, v in acc.items()}, "top": []})
    return rows


def load_questions() -> list[dict]:
    qs = []
    for name in ("questions.jsonl", "questions-reserve.jsonl"):
        qs += [json.loads(ln) for ln in (ONE / name).read_text("utf-8").splitlines() if ln.strip()]
    return [q for q in qs if q["type"] == "article"]


def build_conditions(text: str, pages: list[str], tables: dict[str, str], contexts: dict[str, str]):
    """name -> (index chunks, searched chunks, owner map or None)."""
    articles = parse_articles(text)
    art = article_chunks(articles, max_len=1000)
    tw = table_whole_chunks(tables)
    small, owner = small_to_big(art + tw, size=SMALL)
    conds: dict[str, tuple[list[Chunk], list[Chunk], list[int] | None]] = {
        "article": (art + tw, art + tw, None),
        "article-noprefix": (strip_prefix(art) + tw, strip_prefix(art) + tw, None),
        "article+context": (
            with_context(art, contexts) + tw,
            with_context(art, contexts) + tw,
            None,
        ),
        "page": (page_chunks(pages, articles) + tw, page_chunks(pages, articles) + tw, None),
        # children are indexed; tables are their own parents, appended after the articles
        "small-to-big": (small, art + tw, owner),
    }
    stats = {
        name: {"n": len(idx), "mean_len": round(statistics.mean(len(c.text) for c in idx))}
        for name, (idx, _, _) in conds.items()
    }
    return conds, stats


def _parent_search(
    fn: Callable[[str, int], list[int]], owner: list[int]
) -> Callable[[str, int], list[int]]:
    def rank(q: str, k: int) -> list[int]:
        return parent_rank(fn(q, 200), owner)[:k]

    return rank


def drift_probe(cache: EmbedCache, chunks: list[Chunk], *, n: int = 20) -> float:
    sample = [c.text for c in chunks[:: max(1, len(chunks) // n)]][:n]
    fresh = np.asarray(ollama.embed(sample, model=EMBED_MODEL), dtype=np.float32)
    cached = np.stack([cache.vecs[EmbedCache.key(t)] for t in sample])
    cos = (fresh * cached).sum(1) / (np.linalg.norm(fresh, axis=1) * np.linalg.norm(cached, axis=1))
    return float(cos.min())


def summarize(raw: list[dict], metrics: list[str]) -> tuple[list[dict], list[dict]]:
    by: dict[tuple, list[dict]] = {}
    for r in raw:
        by.setdefault((r["exp"], r["chunking"], r["retriever"]), []).append(r)
    summary = []
    for (exp, ch, ret), rows in sorted(by.items()):
        for m in metrics:
            mean, lo, hi = bootstrap_ci([r[m] for r in rows])
            summary.append(
                {
                    "exp": exp,
                    "chunking": ch,
                    "retriever": ret,
                    "metric": m,
                    "n": len(rows),
                    "mean": round(mean, 3),
                    "lo": round(lo, 3),
                    "hi": round(hi, 3),
                }
            )
    diffs = []
    for (exp, ch_a, ret), rows_a in sorted(by.items()):
        for (exp_b, ch_b, ret_b), rows_b in sorted(by.items()):
            if exp != exp_b or ret != ret_b or ret == "random" or ch_a >= ch_b:
                continue
            for m in metrics:
                a = [r[m] for r in sorted(rows_a, key=lambda r: r["qid"])]
                b = [r[m] for r in sorted(rows_b, key=lambda r: r["qid"])]
                d, lo, hi = paired_diff_ci(a, b)
                diffs.append(
                    {
                        "exp": exp,
                        "retriever": ret,
                        "metric": m,
                        "a": ch_a,
                        "b": ch_b,
                        "diff": round(d, 3),
                        "lo": round(lo, 3),
                        "hi": round(hi, 3),
                        "separable": lo > 0 or hi < 0,
                    }
                )
    # retriever diffs within the baseline chunking (hybrid vs dense / bm25)
    for ch in {r["chunking"] for r in raw}:
        for ra, rb in (("hybrid", "dense"), ("hybrid", "bm25"), ("dense", "bm25")):
            a_rows = sorted(by.get(("article", ch, ra), []), key=lambda r: r["qid"])
            b_rows = sorted(by.get(("article", ch, rb), []), key=lambda r: r["qid"])
            if not a_rows or not b_rows:
                continue
            for m in metrics:
                d, lo, hi = paired_diff_ci([r[m] for r in a_rows], [r[m] for r in b_rows])
                diffs.append(
                    {
                        "exp": "article",
                        "retriever": f"{ra}−{rb}",
                        "metric": m,
                        "a": ch,
                        "b": ch,
                        "diff": round(d, 3),
                        "lo": round(lo, 3),
                        "hi": round(hi, 3),
                        "separable": lo > 0 or hi < 0,
                    }
                )
    return summary, diffs


def main() -> int:
    setup_logging("rag_lab")
    RESULTS.mkdir(exist_ok=True)
    law_pdf = ONE / "corpus/law.pdf"
    text = extract_text(law_pdf)
    pages = [p.extract_text() or "" for p in PdfReader(str(law_pdf)).pages]
    tables = {p.stem: p.read_text("utf-8") for p in sorted((ONE / "corpus/tables").glob("*.txt"))}
    contexts = load_contexts()
    questions = load_questions()
    conds, stats = build_conditions(text, pages, tables, contexts)
    logger.info(
        "conditions built",
        extra={"ctx": {**{k: v["n"] for k, v in stats.items()}, "contexts": len(contexts)}},
    )
    digest = ollama.model_digest(EMBED_MODEL)
    cache = EmbedCache(ROOT / "data/01-chunking/emb-cache" / f"{digest.split(':')[1][:12]}.npz")
    embed = lambda texts: ollama.embed(texts, model=EMBED_MODEL)  # noqa: E731
    metrics = [f"hit@{K}"] + [f"hit@{b}" for b in BUDGETS]

    raw: list[dict] = []
    for name, (index_chunks, out_chunks, owner) in conds.items():
        dense = DenseIndex.build(index_chunks, embed, cache)
        cache.save()
        bm25 = BM25Index(index_chunks)
        hybrid = HybridIndex(dense, bm25)
        searchers = {"dense": dense.search, "bm25": bm25.search, "hybrid": hybrid.search}
        per = {}
        for ret, fn in searchers.items():
            rank = fn if owner is None else _parent_search(fn, owner)
            per[ret] = evaluate(out_chunks, rank, questions, k=K, budgets=BUDGETS)
        per["random"] = evaluate_random(
            out_chunks, questions, k=K, budgets=BUDGETS, draws=RANDOM_DRAWS, seed=0
        )
        for ret, rows in per.items():
            raw.extend({"exp": "article", "chunking": name, "retriever": ret, **r} for r in rows)
        logger.info(
            "condition done",
            extra={
                "ctx": {
                    "chunking": name,
                    **{
                        ret: round(statistics.mean(r["hit@1000"] for r in per[ret]), 3)
                        for ret in per
                    },
                }
            },
        )
    summary, diffs = summarize(raw, metrics)
    min_cos = drift_probe(cache, conds["article"][0])
    ctx_chars = [len(v) for v in contexts.values()]
    meta = {
        "date": date.today().isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "ollama": ollama.version(),
        "embed_model": EMBED_MODEL,
        "embed_model_digest": digest,
        "context_model": "qwen3:8b",
        "context_model_digest": ollama.model_digest("qwen3:8b"),
        "contexts": {
            "n": len(contexts),
            "mean_chars": round(statistics.mean(ctx_chars)) if ctx_chars else 0,
            "empty": sum(1 for v in contexts.values() if not v),
        },
        "pypdf": pkg_version("pypdf"),
        "law_pdf_sha256": hashlib.sha256(law_pdf.read_bytes()).hexdigest(),
        "n_questions": {"article": len(questions)},
        "k": K,
        "budgets": BUDGETS,
        "random_draws": RANDOM_DRAWS,
        "chunk_stats": stats,
        "drift_probe_min_cosine": min_cos,
        "drift_probe_pass": min_cos >= 0.9999,
    }
    (RESULTS / "raw.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in raw), encoding="utf-8"
    )
    for fname, rows in (("summary.csv", summary), ("diffs.csv", diffs)):
        with (RESULTS / fname).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (RESULTS / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    logger.info("done", extra={"ctx": {"rows": len(raw), "drift_min_cos": min_cos}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
