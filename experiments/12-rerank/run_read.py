"""Reading track: models x rerankers x questions on the `full` cell -> results/read/raw.jsonl

    uv run python experiments/12-rerank/run_read.py [--models ...] [--rerankers none,bge,...]

The evidence is 09's budget_evidence over the order run_rerank.py wrote to results/rerank/raw.jsonl
(so this needs no torch); prompt is 09's P0. Resumable (06's done_keys); drift 2 per cell.
"""

from __future__ import annotations

import argparse
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
RESULTS = HERE / "results/read"
RERANK_RAW = HERE / "results" / "rerank" / "raw.jsonl"
sys.path.insert(0, str(HERE))
from chunkers import Chunk  # noqa: E402
from prompts import SCHEMAS, SYSTEM, user_prompt  # noqa: E402
from run_retrieval import build_cell, is_gold  # noqa: E402

logger = logging.getLogger(__name__)

MODELS = ["qwen3:8b", "gpt-oss:20b"]
THINK: dict[str, bool | str] = {"qwen3:8b": False, "gpt-oss:20b": "low"}
OPTIONS = {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 1024}
BUDGET = 1000
READ_CELL = "full"
READ_RERANKERS = ["none", "bge", "qwen3-rr", "listwise"]
DRIFT_PER_CELL = 2


def budget_evidence(ranked: list[Chunk], gold: dict, *, budget: int = BUDGET) -> tuple[str, bool]:
    """Whole chunks in rank order while they fit; the first chunk cut when none fits (09's)."""
    kept: list[Chunk] = []
    used = 0
    for c in ranked:
        if used + len(c.text) > budget:
            break
        used += len(c.text)
        kept.append(c)
    if not kept and ranked:
        c = ranked[0]
        kept = [Chunk(c.text[:budget], list(c.labels), c.start, c.end, dict(c.meta))]
    text = "\n\n".join(f"【段 {i + 1}】\n{c.text}" for i, c in enumerate(kept))
    return text, any(is_gold(c, gold) for c in kept)


def done_keys(raw: Path) -> set[tuple[str, str, str]]:
    if not raw.exists():
        return set()
    return {
        (r["model"], r["condition"], r["qid"])
        for r in (json.loads(ln) for ln in raw.read_text("utf-8").splitlines() if ln.strip())
    }


def load_orders(
    path: Path = RERANK_RAW, *, cell: str = READ_CELL
) -> dict[tuple[str, str], list[int]]:
    """(reranker, qid) -> chunk ids in final order, from run_rerank's raw rows."""
    out = {}
    for ln in path.read_text("utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            if r["cell"] == cell:
                out[(r["reranker"], r["qid"])] = r["order"]
    return out


def ask(model: str, condition: str, q: dict, chunks: list[Chunk], order: list[int]) -> dict:
    gold = {"doc": q["doc"], "quote": q["quote"]}
    text, has_gold = budget_evidence([chunks[i] for i in order], gold)
    prompt = user_prompt(
        q["question"], qtype=q["type"], evidence=text, label="檢索到的段落，可能來自不同法規"
    )
    row = {
        "model": model,
        "condition": condition,
        "qid": q["id"],
        "type": q["type"],
        "prompt_len": len(prompt),
        "evidence_chars": len(text),
        "evidence_has_gold": has_gold,
    }
    try:
        r = ollama.chat(
            prompt,
            model=model,
            system=SYSTEM,
            format=SCHEMAS[q["type"]],
            think=THINK[model],
            options=OPTIONS,
        )
        row.update(
            content=r["content"],
            thinking_len=len(r["thinking"]),
            eval_count=r["eval_count"],
            duration=r["duration"],
            done_reason=r["done_reason"],
            error="",
        )
    except (OSError, KeyError, ValueError) as exc:
        logger.exception(
            "call failed", extra={"ctx": {"model": model, "condition": condition, "qid": q["id"]}}
        )
        row.update(
            content="",
            thinking_len=0,
            eval_count=0,
            duration=0.0,
            done_reason="error",
            error=repr(exc),
        )
    return row


def main(argv: list[str]) -> int:
    from run_rerank import CELLS, load_docs, load_questions, load_texts

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--rerankers", default=",".join(READ_RERANKERS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args(argv)
    models, conditions = args.models.split(","), args.rerankers.split(",")
    setup_logging("rag_lab")
    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.out / "raw.jsonl"
    docs, questions = load_docs(), load_questions()
    if args.limit:
        questions = questions[: args.limit]
    route, chunking, prefix = CELLS[READ_CELL]
    chunks = build_cell(load_texts(docs)[route], docs, chunking=chunking, prefix=prefix)
    orders = load_orders()
    done = done_keys(raw)
    todo = [
        (m, c, q)
        for m in models
        for c in conditions
        for q in questions
        if (m, c, q["id"]) not in done and (c, q["id"]) in orders
    ]
    logger.info("grid", extra={"ctx": {"todo": len(todo), "orders": len(orders)}})
    with raw.open("a", encoding="utf-8") as f:
        for i, (m, c, q) in enumerate(todo, 1):
            row = ask(m, c, q, chunks, orders[(c, q["id"])])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(todo):
                logger.info("progress", extra={"ctx": {"done": i, "todo": len(todo)}})
    first = {
        (r["model"], r["condition"], r["qid"]): r["content"]
        for r in map(json.loads, raw.read_text("utf-8").splitlines())
        if r
    }
    probes = []
    for m in models:
        for c in conditions:
            for q in questions[:DRIFT_PER_CELL]:
                if (c, q["id"]) not in orders:
                    continue
                again = ask(m, c, q, chunks, orders[(c, q["id"])])
                probes.append(
                    {
                        "model": m,
                        "condition": c,
                        "qid": q["id"],
                        "same": again["content"] == first.get((m, c, q["id"])),
                    }
                )
    same = sum(p["same"] for p in probes)
    logger.info("drift", extra={"ctx": {"same": same, "n": len(probes)}})
    meta = {
        "date": date.today().isoformat(),
        "machine": platform.node(),
        "ollama": ollama.version(),
        "models": {m: ollama.model_digest(m) for m in models},
        "think": {m: THINK[m] for m in models},
        "options": OPTIONS,
        "budget": BUDGET,
        "cell": list(CELLS[READ_CELL]),
        "conditions": conditions,
        "n_questions": len(questions),
        "drift": {"same": same, "n": len(probes), "rows": probes},
    }
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
