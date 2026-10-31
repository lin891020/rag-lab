"""Retrieval track: every cell × 160 questions -> results/retrieval/{rows.jsonl, summary.csv}.
No chat model: bge-m3 through 01's embedding cache (new texts are embedded once by the host
ollama), the reranker on torch, the sub-queries read from subqueries.jsonl (decompose.py).

    uv run python experiments/17-table-retrieval/run_retrieval.py [--cells a,b] [--no-rerank]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from rag_lab import ollama
from rag_lab.log import setup_logging

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from evidence import CELLS, Evidence  # noqa: E402
from retrievers import EmbedCache  # noqa: E402
from score import bootstrap_ci  # noqa: E402

logger = logging.getLogger(__name__)

QUESTIONS = HERE / "questions.jsonl"
SUBQUERIES = HERE / "results" / "subqueries.jsonl"
RESULTS = HERE / "results" / "retrieval"
EMB_CACHE = ROOT / "data/01-chunking/emb-cache"  # 16's cache (same model, vectors keyed by text)
EMBED_MODEL = "bge-m3"
RERANKER = "BAAI/bge-reranker-v2-m3"
CHAINS = ["C1", "C2", "C3", "C4", "C5"]
METRICS = ["rows_all_hit", "tables_all_hit", "article_hit"]


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text("utf-8").splitlines() if ln.strip()]


def load_subqueries(path: Path = SUBQUERIES) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    out = {}
    for ln in path.read_text("utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            out[r["qid"]] = r["queries"]
    return out


def embedder():
    digest = ollama.model_digest(EMBED_MODEL)
    cache = EmbedCache(EMB_CACHE / f"{digest.split(':')[1][:12]}.npz")
    return (lambda texts: ollama.embed(texts, model=EMBED_MODEL)), cache


def rows_for(ev: Evidence, cells: list[str], questions: list[dict]) -> list[dict]:
    out = []
    for cell in cells:
        for q in questions:
            if q["expect"]["kind"] == "abstain":  # gold rows of abstain questions are blanks
                continue
            text, hit = ev.for_question(cell, q)
            out.append(
                {
                    "cell": cell,
                    "qid": q["id"],
                    "chain": q["chain"],
                    "cluster": q["cluster"],
                    "chars": len(text),
                    **hit,
                    "rows_all_hit": hit["rows_hit"] == hit["rows_total"],
                    "tables_all_hit": hit["tables_hit"] == hit["tables_total"],
                }
            )
    return out


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for cell in dict.fromkeys(r["cell"] for r in rows):
        sub_cell = [r for r in rows if r["cell"] == cell]
        for chain in ["all", *CHAINS]:
            sub = sub_cell if chain == "all" else [r for r in sub_cell if r["chain"] == chain]
            for m in METRICS:
                mean, lo, hi = bootstrap_ci([float(r[m]) for r in sub])
                out.append(
                    {
                        "cell": cell,
                        "chain": chain,
                        "metric": m,
                        "n": len(sub),
                        "mean": round(mean, 4),
                        "lo": round(lo, 4),
                        "hi": round(hi, 4),
                    }
                )
            out.append(
                {
                    "cell": cell,
                    "chain": chain,
                    "metric": "chars",
                    "n": len(sub),
                    "mean": round(sum(r["chars"] for r in sub) / len(sub), 1),
                    "lo": "",
                    "hi": "",
                }
            )
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args(argv)
    setup_logging("rag_lab")
    cells = args.cells.split(",")
    questions = load_questions()
    subq = load_subqueries()
    if "decomp" in cells and not subq:
        logger.warning(
            "no subqueries yet, decomp skipped", extra={"ctx": {"path": str(SUBQUERIES)}}
        )
        cells = [c for c in cells if c != "decomp"]
    reranker = None
    if "cards+rr" in cells:
        from rerankers import CrossEncoderReranker

        reranker = CrossEncoderReranker(RERANKER)
    embed, cache = embedder()
    ev = Evidence(
        questions, cells=cells, embed=embed, cache=cache, subqueries=subq, reranker=reranker
    )
    rows = rows_for(ev, cells, questions)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rows.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8"
    )
    summary = summarize(rows)
    with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        if s["chain"] == "all" and s["metric"] in ("rows_all_hit", "chars"):
            logger.info("cell", extra={"ctx": {k: s[k] for k in ("cell", "metric", "mean")}})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
