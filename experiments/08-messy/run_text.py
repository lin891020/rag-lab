"""Text track: 01's retrieval harness on the messy variants -> results/text/.

    uv run python experiments/08-messy/run_text.py

Variants (corpus/text/, built by build_corpus.py):
  clean               law.pdf, pypdf                                  blind baseline (= 07's article .80)
  header-footer       title / date / page number stamped on every page, pypdf
  two-column/pypdf    re-typeset in two columns, pypdf (content-stream order: columns stay apart)
  two-column/plumber  the same PDF, pdfplumber (y-sorted: the two columns interleave)
  concat              law.pdf + 輸配電設備裝置規則 (J0030019), pypdf
Chunkings: article (01's one chunk per 條, header-prefixed) and fixed-1000 (the fallback when
the parse breaks). Labels never come from the parse: labels.py marks a chunk with every article
that has a witness line inside it. 01's 109 table chunks are always added (as 07 did).
Retrievers: dense (bge-m3), bm25 (char bigrams), random (noise floor). Metric: hit@1000.
"""

from __future__ import annotations

import csv
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

sys.path.insert(0, str(Path(__file__).parent))
from chunkers import (  # noqa: E402
    Chunk,
    article_chunks,
    fixed_chunks,
    parse_articles,
    table_whole_chunks,
)
from labels import label_chunks, line_map_from_text  # noqa: E402
from metrics import bootstrap_ci, hit_at_budget, hit_at_k, paired_diff_ci  # noqa: E402
from retrievers import BM25Index, DenseIndex, EmbedCache, random_rankings  # noqa: E402

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
ONE = HERE.parent / "01-chunking"
ROOT = HERE.parents[1]
CORPUS = HERE / "corpus"
RESULTS = HERE / "results/text"
EMBED_MODEL = "bge-m3"
SEPS = ["\n\n", "\n", ". ", " ", ""]
K = 5
BUDGETS = [1000, 2000]
RANDOM_DRAWS = 200
RETRIEVERS = ["dense", "bm25", "random"]
CHUNKINGS = ["article", "fixed-1000"]
# variant -> (text file, line map: "clean" = from clean.txt, "typeset" = two-column.lines.jsonl)
VARIANTS: dict[str, tuple[str, str]] = {
    "clean": ("text/clean.txt", "clean"),
    "header-footer": ("text/header-footer.txt", "clean"),
    "two-column/pypdf": ("text/two-column.pypdf.txt", "typeset"),
    "two-column/plumber": ("text/two-column.plumber.txt", "typeset"),
    "concat": ("text/concat.txt", "clean"),
}


def load_questions() -> list[dict]:
    return [
        json.loads(ln)
        for ln in (HERE / "questions-text.jsonl").read_text("utf-8").splitlines()
        if ln.strip()
    ]


def load_line_maps() -> dict[str, list[tuple[str, str]]]:
    clean = line_map_from_text((CORPUS / "text/clean.txt").read_text("utf-8"))
    typeset = [
        tuple(json.loads(ln))
        for ln in (CORPUS / "two-column.lines.jsonl").read_text("utf-8").splitlines()
        if ln.strip()
    ]
    return {"clean": clean, "typeset": typeset}  # type: ignore[dict-item]


def build_conditions(
    text: str, line_map: list[tuple[str, str]], tables: dict[str, str]
) -> tuple[dict[str, list[Chunk]], dict]:
    """chunking name -> chunks (text chunks relabelled by witnesses + 01's table chunks)."""
    articles = parse_articles(text)
    spans = [(a.start, a.end, str(a.num)) for a in articles]
    tw = table_whole_chunks(tables)
    raw = {
        "article": article_chunks(articles, max_len=1000),
        "fixed-1000": fixed_chunks(text, size=1000, overlap=0, separators=SEPS, spans=spans),
    }
    conds: dict[str, list[Chunk]] = {}
    stats: dict = {"articles_parsed": len(articles), "chars": len(text)}
    for name, chunks in raw.items():
        relabelled = label_chunks(chunks, line_map)
        conds[name] = relabelled + tw
        stats[name] = {
            "n": len(relabelled),
            "mean_len": round(statistics.mean(len(c.text) for c in relabelled)),
            "unlabelled": sum(1 for c in relabelled if not c.labels),
            "parse_labels_agree": round(
                sum(set(a.labels) == set(b.labels) for a, b in zip(chunks, relabelled, strict=True))
                / max(1, len(chunks)),
                3,
            ),
        }
    return conds, stats


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


def drift_probe(cache: EmbedCache, chunks: list[Chunk], *, n: int = 20) -> float:
    sample = [c.text for c in chunks[:: max(1, len(chunks) // n)]][:n]
    fresh = np.asarray(ollama.embed(sample, model=EMBED_MODEL), dtype=np.float32)
    cached = np.stack([cache.vecs[EmbedCache.key(t)] for t in sample])
    cos = (fresh * cached).sum(1) / (np.linalg.norm(fresh, axis=1) * np.linalg.norm(cached, axis=1))
    return float(cos.min())


def summarize(raw: list[dict], metrics: list[str]) -> tuple[list[dict], list[dict]]:
    """Summary per (variant, chunking, retriever); paired diffs of every variant against clean."""
    by: dict[tuple, list[dict]] = {}
    for r in raw:
        by.setdefault((r["variant"], r["chunking"], r["retriever"]), []).append(r)
    summary = []
    for (variant, ch, ret), rows in sorted(by.items()):
        for m in metrics:
            mean, lo, hi = bootstrap_ci([r[m] for r in rows])
            summary.append(
                {
                    "variant": variant,
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
    for (variant, ch, ret), rows_v in sorted(by.items()):
        base = by.get(("clean", ch, ret))
        if variant == "clean" or ret == "random" or not base:
            continue
        for m in metrics:
            a = [r[m] for r in sorted(rows_v, key=lambda r: r["qid"])]
            b = [r[m] for r in sorted(base, key=lambda r: r["qid"])]
            d, lo, hi = paired_diff_ci(a, b)
            diffs.append(
                {
                    "variant": variant,
                    "chunking": ch,
                    "retriever": ret,
                    "metric": m,
                    "diff": round(d, 3),
                    "lo": round(lo, 3),
                    "hi": round(hi, 3),
                    "separable": lo > 0 or hi < 0,
                }
            )
    return summary, diffs


def main() -> int:
    setup_logging("rag_lab")
    RESULTS.mkdir(parents=True, exist_ok=True)
    tables = {p.stem: p.read_text("utf-8") for p in sorted((ONE / "corpus/tables").glob("*.txt"))}
    line_maps = load_line_maps()
    questions = load_questions()
    digest = ollama.model_digest(EMBED_MODEL)
    cache = EmbedCache(ROOT / "data/01-chunking/emb-cache" / f"{digest.split(':')[1][:12]}.npz")
    embed = lambda texts: ollama.embed(texts, model=EMBED_MODEL)  # noqa: E731
    metrics = [f"hit@{K}"] + [f"hit@{b}" for b in BUDGETS]

    raw: list[dict] = []
    all_stats: dict = {}
    for variant, (fname, lm) in VARIANTS.items():
        text = (CORPUS / fname).read_text("utf-8")
        conds, stats = build_conditions(text, line_maps[lm], tables)
        all_stats[variant] = stats
        logger.info(
            "variant",
            extra={
                "ctx": {
                    "variant": variant,
                    **{k: v for k, v in stats.items() if not isinstance(v, dict)},
                    **{f"{k}_n": v["n"] for k, v in stats.items() if isinstance(v, dict)},
                }
            },
        )
        for ch, chunks in conds.items():
            dense = DenseIndex.build(chunks, embed, cache)
            cache.save()
            bm25 = BM25Index(chunks)
            per = {
                "dense": evaluate(chunks, dense.search, questions, k=K, budgets=BUDGETS),
                "bm25": evaluate(chunks, bm25.search, questions, k=K, budgets=BUDGETS),
                "random": evaluate_random(
                    chunks, questions, k=K, budgets=BUDGETS, draws=RANDOM_DRAWS, seed=0
                ),
            }
            for ret, rows in per.items():
                raw.extend(
                    {"variant": variant, "chunking": ch, "retriever": ret, **r} for r in rows
                )
            logger.info(
                "condition done",
                extra={
                    "ctx": {
                        "variant": variant,
                        "chunking": ch,
                        **{
                            ret: round(statistics.mean(r["hit@1000"] for r in per[ret]), 3)
                            for ret in per
                        },
                    }
                },
            )
    summary, diffs = summarize(raw, metrics)
    clean_text = (CORPUS / VARIANTS["clean"][0]).read_text("utf-8")
    probe_chunks, _ = build_conditions(clean_text, line_maps["clean"], tables)
    min_cos = drift_probe(cache, probe_chunks["article"])
    corpus_stats = json.loads((CORPUS / "stats.json").read_text("utf-8"))
    meta = {
        "date": date.today().isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "ollama": ollama.version(),
        "embed_model": EMBED_MODEL,
        "embed_model_digest": digest,
        "pypdf": pkg_version("pypdf"),
        "pdfplumber": pkg_version("pdfplumber"),
        "pymupdf": pkg_version("pymupdf"),
        "law_pdf_sha256": corpus_stats["text"]["law_pdf_sha256"],
        "build": corpus_stats["text"],
        "n_questions": len(questions),
        "k": K,
        "budgets": BUDGETS,
        "random_draws": RANDOM_DRAWS,
        "variants": all_stats,
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
