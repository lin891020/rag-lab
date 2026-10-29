"""Retrieval track: dense top-50 -> each reranker -> rank/budget metrics per question.

    uv run --group rerank python experiments/12-rerank/run_rerank.py [--cells full,fixed]
        [--rerankers none,bm25,bge,qwen3-rr,listwise] [--limit N]

Writes results/rerank/raw.jsonl (one row per cell x reranker x question: the reranked chunk ids,
seconds, metrics), summary.csv, diffs.csv (each reranker − none), meta.json. run_read.py rebuilds
the same chunks and reads the orders from raw.jsonl, so the reading track needs no reranker model.
Resumable per row; cross-encoder scores are cached in data/12-rerank/.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import sys
from datetime import date
from pathlib import Path

from rag_lab import ollama
from rag_lab.log import setup_logging

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
CORPUS09 = HERE.parent / "09-messy-real" / "corpus"
CORPUS10 = HERE.parent / "10-fix-ablation" / "corpus"
RESULTS = HERE / "results" / "rerank"
DATA = ROOT / "data/09-messy-real"
ROUTES = {"base": CORPUS09, "ocr": CORPUS09, "rows": CORPUS09, "ocrrows": CORPUS10}
sys.path.insert(0, str(HERE))
from chunkers import Chunk  # noqa: E402
from metrics import bootstrap_ci, paired_diff_ci  # noqa: E402
from rerankers import CrossEncoderReranker, ListwiseReranker, Qwen3Reranker, rrf  # noqa: E402
from retrievers import BM25Index, DenseIndex, EmbedCache  # noqa: E402
from run_read import budget_evidence  # noqa: E402
from run_retrieval import DEPTH, EMBED_MODEL, build_cell, first_hit_rank, hit_at_budget, hit_at_k  # noqa: E402

logger = logging.getLogger(__name__)

CELLS: dict[str, tuple[str, str, bool]] = {
    "full": ("ocrrows", "article-cn+pre", True),
    "fixed": ("ocrrows", "fixed-1000", True),
}
RERANKERS = ["none", "bm25", "bge", "qwen3-rr", "listwise"]
BGE = "BAAI/bge-reranker-v2-m3"
LISTWISE_MODEL = "qwen3:8b"
METRICS = ["hit@1", "mrr", "hit@5", "hit@1000", "hit@2000", "budget_has_gold"]
DRIFT_PER_CELL = 2


def metrics_for(ranked: list[Chunk], gold: dict) -> dict:
    rank = first_hit_rank(ranked, gold)
    return {
        "hit@1": rank == 0,
        "mrr": 0.0 if rank < 0 else 1.0 / (rank + 1),
        "hit@5": hit_at_k(ranked, gold, k=5),
        "hit@1000": hit_at_budget(ranked, gold, budget=1000),
        "hit@2000": hit_at_budget(ranked, gold, budget=2000),
        "budget_has_gold": budget_evidence(ranked, gold)[1],
    }


def load_docs() -> list[dict]:
    return json.loads((CORPUS09 / "manifest.json").read_text("utf-8"))["docs"]


def load_questions() -> list[dict]:
    path = CORPUS09.parent / "questions.jsonl"
    return [json.loads(ln) for ln in path.read_text("utf-8").splitlines() if ln.strip()]


def load_texts(docs: list[dict]) -> dict[str, dict[str, str]]:
    return {
        route: {
            d["sysid"]: (folder / "text" / f"{d['sysid']}.{route}.txt").read_text("utf-8")
            for d in docs
        }
        for route, folder in ROUTES.items()
    }


def make_reranker(name: str):
    if name == "bge":
        return CrossEncoderReranker(BGE)
    if name == "qwen3-rr":
        return Qwen3Reranker()
    if name == "listwise":
        return ListwiseReranker(LISTWISE_MODEL)
    return None  # none / bm25 are handled inline


class Cell:
    def __init__(self, name: str, texts, docs, embed, cache):
        route, chunking, prefix = CELLS[name]
        self.name = name
        self.chunks = build_cell(texts[route], docs, chunking=chunking, prefix=prefix)
        self.dense = DenseIndex.build(self.chunks, embed, cache)
        self.bm25 = BM25Index(self.chunks)
        cache.save()
        logger.info("cell built", extra={"ctx": {"cell": name, "chunks": len(self.chunks)}})

    def candidates(self, question: str) -> list[int]:
        return self.dense.search(question, DEPTH)

    def fused(self, question: str) -> list[int]:
        d = self.dense.search(question, len(self.chunks))
        b = self.bm25.search(question, len(self.chunks))
        return rrf([d, b], depth=len(self.chunks))[:DEPTH]


def rerank_one(cell: Cell, reranker: str, rr, q: dict) -> tuple[list[int], float]:
    """Chunk ids in final order and the reranker's seconds."""
    if reranker == "bm25":
        return cell.fused(q["question"]), 0.0
    cand = cell.candidates(q["question"])
    if reranker == "none" or rr is None:
        return cand, 0.0
    order, seconds = rr.rerank(q["question"], [cell.chunks[i] for i in cand])
    return [cand[i] for i in order], seconds


def summarize(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    cells: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        cells.setdefault((r["cell"], r["reranker"]), []).append(r)
    summary, diffs = [], []
    for (cell, rr), rs in sorted(cells.items()):
        rs.sort(key=lambda r: r["qid"])
        for m in METRICS:
            mean, lo, hi = bootstrap_ci([float(r[m]) for r in rs])
            summary.append(
                {
                    "cell": cell,
                    "reranker": rr,
                    "metric": m,
                    "mean": mean,
                    "lo": lo,
                    "hi": hi,
                    "n": len(rs),
                }
            )
        summary.append(
            {
                "cell": cell,
                "reranker": rr,
                "metric": "seconds",
                "mean": sum(r["seconds"] for r in rs) / len(rs),
                "lo": "",
                "hi": "",
                "n": len(rs),
            }
        )
        base = cells.get((cell, "none"))
        if rr == "none" or not base:
            continue
        qa = {r["qid"]: r for r in rs}
        qb = {r["qid"]: r for r in base}
        common = sorted(set(qa) & set(qb))
        for m in METRICS:
            d, lo, hi = paired_diff_ci(
                [float(qa[k][m]) for k in common], [float(qb[k][m]) for k in common]
            )
            diffs.append(
                {
                    "cell": cell,
                    "reranker": rr,
                    "metric": m,
                    "diff": d,
                    "lo": lo,
                    "hi": hi,
                    "separable": lo > 0 or hi < 0,
                    "n": len(common),
                }
            )
    return summary, diffs


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--rerankers", default=",".join(RERANKERS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args(argv)
    setup_logging("rag_lab")
    args.out.mkdir(parents=True, exist_ok=True)
    raw_path = args.out / "raw.jsonl"
    docs, questions = load_docs(), load_questions()
    if args.limit:
        questions = questions[: args.limit]
    texts = load_texts(docs)
    digest = ollama.model_digest(EMBED_MODEL)
    cache = EmbedCache(DATA / "emb-cache" / f"{digest.split(':')[1][:12]}.npz")
    embed = lambda t: ollama.embed(t, model=EMBED_MODEL)  # noqa: E731
    done = set()
    rows: list[dict] = []
    if raw_path.exists():
        rows = [json.loads(ln) for ln in raw_path.read_text("utf-8").splitlines() if ln.strip()]
        done = {(r["cell"], r["reranker"], r["qid"]) for r in rows}
    drift = []
    with raw_path.open("a", encoding="utf-8") as f:
        for cell_name in args.cells.split(","):
            cell = Cell(cell_name, texts, docs, embed, cache)
            for name in args.rerankers.split(","):
                rr = make_reranker(name)
                n_new = 0
                for q in questions:
                    if (cell_name, name, q["id"]) in done:
                        continue
                    gold = {"doc": q["doc"], "quote": q["quote"]}
                    ids, seconds = rerank_one(cell, name, rr, q)
                    row = {
                        "cell": cell_name,
                        "reranker": name,
                        "qid": q["id"],
                        "type": q["type"],
                        "order": ids,
                        "seconds": seconds,
                        **metrics_for([cell.chunks[i] for i in ids], gold),
                    }
                    rows.append(row)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    n_new += 1
                if rr is not None:
                    rr.save()
                    for q in questions[:DRIFT_PER_CELL]:
                        again, _ = rerank_one(cell, name, rr, q)
                        first = next(
                            r["order"]
                            for r in rows
                            if (r["cell"], r["reranker"], r["qid"]) == (cell_name, name, q["id"])
                        )
                        drift.append(
                            {
                                "cell": cell_name,
                                "reranker": name,
                                "qid": q["id"],
                                "same": again == first,
                            }
                        )
                logger.info(
                    "reranked",
                    extra={
                        "ctx": {
                            "cell": cell_name,
                            "reranker": name,
                            "new": n_new,
                            "bad_outputs": getattr(rr, "bad_outputs", 0),
                        }
                    },
                )
    summary, diffs = summarize(rows)
    _write_csv(args.out / "summary.csv", summary)
    if diffs:
        _write_csv(args.out / "diffs.csv", diffs)
    meta = {
        "date": date.today().isoformat(),
        "machine": platform.node(),
        "ollama": ollama.version(),
        "embed": {EMBED_MODEL: digest},
        "listwise_model": {LISTWISE_MODEL: ollama.model_digest(LISTWISE_MODEL)},
        "bge": BGE,
        "qwen3_reranker": Qwen3Reranker.NAME,
        "depth": DEPTH,
        "cells": {k: list(v) for k, v in CELLS.items()},
        "n_questions": len(questions),
        "drift": {"same": sum(d["same"] for d in drift), "n": len(drift), "rows": drift},
    }
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    logger.info("done", extra={"ctx": {"rows": len(rows)}})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
