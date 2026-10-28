"""Run the whole grid and write results/. Idempotent: embeddings are cached under data/.

    uv run python experiments/01-chunking/run.py

Two experiments share one question file:
  article questions: text chunking varies (fixed1000 / article / fixedN), tables fixed as table-whole
  table questions:   text fixed as article, table chunking varies (whole / row / flat-fixed)
Each condition is searched by dense (bge-m3), bm25 (char bigrams) and random (noise floor).
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

from rag_lab import ollama
from rag_lab.log import setup_logging
from rag_lab.pdf import extract_text

sys.path.insert(0, str(Path(__file__).parent))
from chunkers import (  # noqa: E402
    Chunk,
    article_chunks,
    fixed_chunks,
    parse_articles,
    table_flat_fixed_chunks,
    table_row_chunks,
    table_whole_chunks,
)
from metrics import bootstrap_ci, hit_at_budget, hit_at_k, paired_diff_ci  # noqa: E402
from retrievers import BM25Index, DenseIndex, EmbedCache, random_rankings  # noqa: E402

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
EMBED_MODEL = "bge-m3"
SEPS = ["\n\n", "\n", ". ", " ", ""]  # securag's splitter separators
K = 5
BUDGETS = [1000, 2000]
RANDOM_DRAWS = 200
RETRIEVERS = ["dense", "bm25", "random"]


def evaluate(
    chunks: list[Chunk],
    rank: Callable[[str, int], list[int]],
    questions: list[dict],
    *,
    k: int,
    budgets: list[int],
) -> list[dict]:
    """One row per question: hit@k and hit@<budget> for every budget, plus the top labels."""
    rows = []
    depth = max(k, 50)  # budgets may reach past k chunks when chunks are small
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
    """Noise floor: the same metrics averaged over `draws` random rankings."""
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


def build_conditions(text: str, tables: dict[str, str]) -> tuple[dict, dict, dict]:
    articles = parse_articles(text)
    spans = [(a.start, a.end, str(a.num)) for a in articles]
    art = article_chunks(articles, max_len=1000)
    size_match = round(statistics.mean(len(c.text) for c in art))
    text_strats = {
        "fixed1000": fixed_chunks(text, size=1000, overlap=200, separators=SEPS, spans=spans),
        "article": art,
        f"fixed{size_match}": fixed_chunks(
            text, size=size_match, overlap=0, separators=SEPS, spans=spans
        ),
    }
    table_strats = {
        "table-whole": table_whole_chunks(tables),
        "table-row": table_row_chunks(tables),
        "table-flat-fixed": table_flat_fixed_chunks(
            tables, size=1000, overlap=200, separators=SEPS
        ),
    }
    conditions = {
        "article": {name: cs + table_strats["table-whole"] for name, cs in text_strats.items()},
        "table": {name: art + cs for name, cs in table_strats.items()},
    }
    stats = {
        name: {"n": len(cs), "mean_len": round(statistics.mean(len(c.text) for c in cs))}
        for name, cs in {**text_strats, **table_strats}.items()
    }
    return conditions, stats, {"size_match": size_match}


def drift_probe(cache: EmbedCache, chunks: list[Chunk], *, n: int = 20) -> float:
    """Re-embed n chunks straight from ollama and return the minimum cosine vs the cache."""
    sample = [c.text for c in chunks[:: max(1, len(chunks) // n)]][:n]
    fresh = np.asarray(ollama.embed(sample, model=EMBED_MODEL), dtype=np.float32)
    cached = np.stack([cache.vecs[EmbedCache.key(t)] for t in sample])
    cos = (fresh * cached).sum(1) / (np.linalg.norm(fresh, axis=1) * np.linalg.norm(cached, axis=1))
    return float(cos.min())


def summarize(raw: list[dict], metrics: list[str]) -> tuple[list[dict], list[dict]]:
    """Per (exp, chunking, retriever, metric): mean and CI. Plus paired diffs between chunkings."""
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
    return summary, diffs


def main() -> int:
    setup_logging("rag_lab")
    RESULTS.mkdir(exist_ok=True)
    law_pdf = HERE / "corpus/law.pdf"
    text = extract_text(law_pdf)
    tables = {
        p.stem: p.read_text(encoding="utf-8")
        for p in sorted((HERE / "corpus/tables").glob("*.txt"))
    }
    questions = [
        json.loads(ln)
        for ln in (HERE / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    conditions, stats, extra = build_conditions(text, tables)
    logger.info("conditions built", extra={"ctx": {k: v["n"] for k, v in stats.items()}})

    digest = ollama.model_digest(EMBED_MODEL)
    cache = EmbedCache(ROOT / "data/01-chunking/emb-cache" / f"{digest.split(':')[1][:12]}.npz")
    embed = lambda texts: ollama.embed(texts, model=EMBED_MODEL)  # noqa: E731
    metrics = [f"hit@{K}"] + [f"hit@{b}" for b in BUDGETS]

    raw: list[dict] = []
    for exp, strategies in conditions.items():
        qs = [q for q in questions if q["type"] == exp]
        for name, chunks in strategies.items():
            dense = DenseIndex.build(chunks, embed, cache)
            cache.save()
            bm25 = BM25Index(chunks)
            per = {
                "dense": evaluate(chunks, dense.search, qs, k=K, budgets=BUDGETS),
                "bm25": evaluate(chunks, bm25.search, qs, k=K, budgets=BUDGETS),
                "random": evaluate_random(
                    chunks, qs, k=K, budgets=BUDGETS, draws=RANDOM_DRAWS, seed=0
                ),
            }
            for ret, rows in per.items():
                raw.extend({"exp": exp, "chunking": name, "retriever": ret, **r} for r in rows)
            logger.info(
                "condition done",
                extra={
                    "ctx": {
                        "exp": exp,
                        "chunking": name,
                        "dense_hit5": statistics.mean(r[f"hit@{K}"] for r in per["dense"]),
                    }
                },
            )

    summary, diffs = summarize(raw, metrics)
    min_cos = drift_probe(cache, conditions["article"]["article"])
    meta = {
        "date": date.today().isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "ollama": ollama.version(),
        "embed_model": EMBED_MODEL,
        "embed_model_digest": digest,
        "pypdf": pkg_version("pypdf"),
        "pdfplumber": pkg_version("pdfplumber"),
        "law_pdf_sha256": hashlib.sha256(law_pdf.read_bytes()).hexdigest(),
        "n_questions": {
            "article": sum(q["type"] == "article" for q in questions),
            "table": sum(q["type"] == "table" for q in questions),
        },
        "k": K,
        "budgets": BUDGETS,
        "random_draws": RANDOM_DRAWS,
        "chunk_stats": stats,
        "drift_probe_min_cosine": min_cos,
        "drift_probe_pass": min_cos >= 0.9999,
        **extra,
    }
    (RESULTS / "raw.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in raw), encoding="utf-8"
    )
    for name, rows in (("summary.csv", summary), ("diffs.csv", diffs)):
        with (RESULTS / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (RESULTS / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "written",
        extra={
            "ctx": {
                "raw": len(raw),
                "summary": len(summary),
                "diffs": len(diffs),
                "drift_min_cos": min_cos,
            }
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
