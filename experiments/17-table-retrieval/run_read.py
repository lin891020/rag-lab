"""Reading track: the new cells × 2 models × 160 questions against the host ollama
-> results/read/raw.jsonl (resumable). 16's L1 prompt (the model fills `value` itself, no tool);
evidence from evidence.py. 16's L1-struct and L1-oracle rows are copied in as struct@1500 and
oracle (same prompt, same evidence: tests/test_run_read.py proves the evidence text) and checked
by re-asking 10 spread questions per (model, cell); new cells get 2 drift probes each.

    uv run python experiments/17-table-retrieval/run_read.py [--models a,b] [--limit N]
        [--cells struct@3000,cards] [--out results/read] [--retry-only] [--redo]
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from collections.abc import Iterator
from datetime import date
from pathlib import Path

from rag_lab import ollama
from rag_lab.log import setup_logging

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from evidence import BUDGETS, Evidence  # noqa: E402
from prompts import SCHEMA, SYSTEM, user_prompt  # noqa: E402
from run_retrieval import embedder, load_questions, load_subqueries  # noqa: E402

logger = logging.getLogger(__name__)

RESULTS = HERE / "results/read"
RAW16 = HERE.parent / "16-rule-lookup-compute/results/read/raw.jsonl"
META16 = HERE.parent / "16-rule-lookup-compute/results/read/meta.json"
MODELS = ["qwen3:8b", "gpt-oss:20b"]
THINK: dict[str, bool | str] = {"qwen3:8b": False, "gpt-oss:20b": "low"}  # see 02 README
OPTIONS = {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 1024}
NEW_CELLS = ["struct@3000-nosub", "struct@3000", "cards", "decomp", "decomp@4000"]
REUSED = {"L1-struct": "struct@1500", "L1-oracle": "oracle"}  # 16's name -> 17's
PROBES_NEW = 2
PROBES_REUSED = 10
LABELS = {
    "oracle": "規定與附表節錄",
    "retrieved": "檢索到的段落，可能來自不同條文與附表",
}


def grid(models: list[str], cells: list[str], questions: list[dict]) -> Iterator[tuple]:
    for m in models:
        for c in cells:
            for q in questions:
                yield (m, c, q["id"])


def read_raw(raw: Path) -> list[dict]:
    """Rows of a jsonl file; a last line cut off by a kill mid-write is skipped and logged."""
    if not raw.exists():
        return []
    lines = [ln for ln in raw.read_text("utf-8").splitlines() if ln.strip()]
    out = []
    for i, ln in enumerate(lines):
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            logger.warning("skipped a truncated last line", extra={"ctx": {"path": str(raw)}})
    return out


def done_keys(rows: list[dict]) -> set[tuple[str, str, str]]:
    return {(r["model"], r["condition"], r["qid"]) for r in rows if not r.get("error")}


def reused_rows(path: Path = RAW16) -> list[dict]:
    """16's L1-struct / L1-oracle rows renamed to 17's cells."""
    out = []
    for r in read_raw(path):
        if r["condition"] in REUSED:
            out.append({**r, "condition": REUSED[r["condition"]], "reused_from": "16"})
    return out


def missing_reused(have: list[dict], models: list[str], qids: set[str]) -> list[dict]:
    """16's rows for (model, cell, qid) keys not yet in `have` (copied by key, so a partial
    earlier copy is completed)."""
    keys = {(r["model"], r["condition"], r["qid"]) for r in have if r.get("reused_from")}
    return [
        r
        for r in reused_rows()
        if r["model"] in models
        and r["qid"] in qids
        and (r["model"], r["condition"], r["qid"]) not in keys
    ]


def check_digests(models: list[str]) -> dict[str, str]:
    """The chat models must be the ones 16 ran: its rows are reused."""
    meta16 = json.loads(META16.read_text("utf-8"))
    digests = {m: ollama.model_digest(m) for m in models}
    bad = {
        m: (d, meta16["models"].get(m)) for m, d in digests.items() if d != meta16["models"].get(m)
    }
    if bad:
        raise SystemExit(f"model digests differ from 16: {bad}")
    return digests


def spread(questions: list[dict], n: int) -> list[dict]:
    """n questions evenly spread over the list (every chain is represented)."""
    step = max(1, len(questions) // n)
    return questions[::step][:n]


def ask(model: str, cell: str, q: dict, ev: Evidence) -> dict:
    text, hit = ev.for_question(cell, q)
    label = LABELS["oracle"] if cell == "oracle" else LABELS["retrieved"]
    prompt = user_prompt(q["question"], evidence=text, label=label, tool=False)
    row = {
        "model": model,
        "condition": cell,
        "qid": q["id"],
        "chain": q["chain"],
        "prompt_len": len(prompt),
        "evidence_chars": len(text),
        "evidence_text": text,
        **hit,
    }
    try:
        r = ollama.chat(
            prompt, model=model, system=SYSTEM, format=SCHEMA, think=THINK[model], options=OPTIONS
        )
        row.update(
            content=r["content"],
            raw_content=r["content"],
            expr_ok=None,
            thinking_len=len(r["thinking"]),
            eval_count=r["eval_count"],
            duration=r["duration"],
            done_reason=r["done_reason"],
            error="",
        )
    except (OSError, KeyError, ValueError) as exc:
        logger.exception(
            "call failed", extra={"ctx": {"model": model, "condition": cell, "qid": q["id"]}}
        )
        row.update(
            content="",
            raw_content="",
            expr_ok=None,
            thinking_len=0,
            eval_count=0,
            duration=0.0,
            done_reason="error",
            error=repr(exc),
        )
    return row


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cells", default=",".join(NEW_CELLS))
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument(
        "--retry-only", action="store_true", help="redo failed calls; no probes, no meta"
    )
    ap.add_argument(
        "--redo",
        action="store_true",
        help="ask every key again, done or not (a reused cell whose probes failed); the new rows "
        "are appended and score.py keeps the last good row per key",
    )
    args = ap.parse_args(argv)
    models = args.models.split(",")
    cells = args.cells.split(",")
    setup_logging("rag_lab")
    questions = load_questions()
    if args.limit:
        questions = questions[: args.limit]
    byid = {q["id"]: q for q in questions}
    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.out / "raw.jsonl"
    digests = check_digests(models)
    have = read_raw(raw)
    old = missing_reused(have, models, set(byid))
    if old:
        with raw.open("a", encoding="utf-8") as f:
            for r in old:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("copied 16 rows", extra={"ctx": {"n": len(old)}})
        have = read_raw(raw)
    subq = load_subqueries()
    if any(c.startswith("decomp") for c in cells) and set(byid) - set(subq):
        raise SystemExit(f"sub-queries missing for {len(set(byid) - set(subq))} questions")
    embed, cache = embedder()
    ev = Evidence(
        questions,
        cells=[*cells, "struct@1500"],
        embed=embed,
        cache=cache,
        subqueries=subq,
    )
    probes = []
    first16 = {(r["model"], r["condition"], r["qid"]): r for r in have if r.get("reused_from")}

    def probe(m: str, c: str, q: dict, ref: dict | None) -> None:
        again = ask(m, c, q, ev)
        probes.append(
            {
                "model": m,
                "condition": c,
                "qid": q["id"],
                "same": ref is not None and again["raw_content"] == ref["raw_content"],
                "content": again["raw_content"],
                "first": ref["raw_content"] if ref else None,
            }
        )

    for m in [] if args.retry_only else models:  # reused cells first: a mismatch shows early
        for c in REUSED.values():
            for q in spread(questions, PROBES_REUSED):
                probe(m, c, q, first16.get((m, c, q["id"])))
    if probes:
        logger.info(
            "reused probes",
            extra={"ctx": {"same": sum(p["same"] for p in probes), "n": len(probes)}},
        )
    done = set() if args.redo else done_keys(have)
    todo = [k for k in grid(models, cells, questions) if k not in done]
    logger.info("grid", extra={"ctx": {"cells": cells, "todo": len(todo)}})
    cell_secs: dict[tuple[str, str], list[float]] = {}
    with raw.open("a", encoding="utf-8") as f:
        for i, (m, c, qid) in enumerate(todo, 1):
            row = ask(m, c, byid[qid], ev)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            secs = cell_secs.setdefault((m, c), [])
            secs.append(row["duration"])
            if len(secs) == 3:
                mean = sum(secs) / 3
                logger.info(
                    "cell timing",
                    extra={
                        "ctx": {
                            "model": m,
                            "condition": c,
                            "sec_per_call": round(mean, 1),
                            "cell_hours": round(mean * len(questions) / 3600, 2),
                        }
                    },
                )
            if i % 20 == 0 or i == len(todo):
                logger.info(
                    "progress",
                    extra={"ctx": {"done": i, "todo": len(todo), "model": m, "condition": c}},
                )
    if args.retry_only:
        return 0
    first: dict[tuple[str, str, str], dict] = {}
    for r in read_raw(raw):  # the last good row per key
        if not r.get("error"):
            first[(r["model"], r["condition"], r["qid"])] = r
    for m in models:
        for c in cells:
            for q in spread(questions, PROBES_NEW):
                probe(m, c, q, first.get((m, c, q["id"])))
    by_cell: dict[str, list[int]] = {}
    for p in probes:
        by_cell.setdefault(f"{p['model']} {p['condition']}", []).append(int(p["same"]))
    logger.info(
        "drift", extra={"ctx": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_cell.items())}}
    )
    meta16 = json.loads(META16.read_text("utf-8"))
    meta = {
        "date": date.today().isoformat(),
        "machine": platform.node(),
        "ollama": ollama.version(),
        "models": digests,
        "models_16": meta16["models"],
        "same_digest_as_16": True,  # checked at start (check_digests)
        "ollama_16": meta16.get("ollama"),
        "think": {m: THINK[m] for m in models},
        "options": OPTIONS,
        "budgets": {c: BUDGETS[c] for c in cells},
        "cells": cells,
        "reused": REUSED,
        "n_questions": len(questions),
        "drift": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_cell.items())},
        "drift_rows": probes,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
