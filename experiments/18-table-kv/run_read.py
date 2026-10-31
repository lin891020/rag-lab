"""Reading track: cards-kv and cards × 2 models × 160 questions against the host ollama
-> results/read/raw.jsonl (resumable). 17's prompt and options. qwen3's cards rows are copied from
17 (byte-identical reruns there; same evidence: tests/test_run_read.py) and checked by re-asking
10 spread questions; when any differs the whole cell is asked again. gpt-oss's cards is asked
again on purpose: 17 showed it does not repeat itself, so this is the new baseline and the noise
measurement at once. New cells get 2 drift probes each. Every row records prompt_eval_count.

    uv run python experiments/18-table-kv/run_read.py [--models a,b] [--limit N]
        [--cells cards-kv,cards] [--out results/read] [--retry-only]
"""

from __future__ import annotations

import argparse
import gzip
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
from run_retrieval import embedder, load_questions  # noqa: E402

logger = logging.getLogger(__name__)

RESULTS = HERE / "results/read"
RAW17 = HERE.parent / "17-table-retrieval/results/read/raw.jsonl.gz"
META17 = HERE.parent / "17-table-retrieval/results/read/meta.json"
MODELS = ["qwen3:8b", "gpt-oss:20b"]
THINK: dict[str, bool | str] = {"qwen3:8b": False, "gpt-oss:20b": "low"}  # see 02 README
OPTIONS = {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 1024}
CELLS = ["cards-kv", "cards"]
REUSED = {("qwen3:8b", "cards")}  # (model, cell) copied from 17
PROBES_NEW = 2
PROBES_REUSED = 10
LABEL = "檢索到的段落，可能來自不同條文與附表"  # 17's label for every retrieved cell


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


def rows17(path: Path = RAW17) -> list[dict]:
    """17's reading rows, the last good row per (model, cell, qid) (what 17's score.py used)."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        rows = [json.loads(ln) for ln in f if ln.strip()]
    last = {}
    for r in rows:
        if not r.get("error"):
            last[(r["model"], r["condition"], r["qid"])] = r
    return list(last.values())


def reused_rows(path: Path = RAW17) -> list[dict]:
    """17's rows for the REUSED (model, cell) pairs, marked reused_from=17."""
    return [
        {**r, "reused_from": "17"} for r in rows17(path) if (r["model"], r["condition"]) in REUSED
    ]


def missing_reused(have: list[dict], models: list[str], qids: set[str]) -> list[dict]:
    """17's rows for keys not yet copied into `have` (so a partial earlier copy is completed)."""
    keys = {(r["model"], r["condition"], r["qid"]) for r in have if r.get("reused_from")}
    return [
        r
        for r in reused_rows()
        if r["model"] in models
        and r["qid"] in qids
        and (r["model"], r["condition"], r["qid"]) not in keys
    ]


def check_digests(models: list[str]) -> dict[str, str]:
    """The chat models must be the ones 17 ran: its rows are reused and compared."""
    meta17 = json.loads(META17.read_text("utf-8"))
    digests = {m: ollama.model_digest(m) for m in models}
    bad = {
        m: (d, meta17["models"].get(m)) for m, d in digests.items() if d != meta17["models"].get(m)
    }
    if bad:
        raise SystemExit(f"model digests differ from 17: {bad}")
    return digests


def spread(questions: list[dict], n: int) -> list[dict]:
    """n questions evenly spread over the list (every chain is represented)."""
    step = max(1, len(questions) // n)
    return questions[::step][:n]


def ask(model: str, cell: str, q: dict, ev: Evidence) -> dict:
    text, hit = ev.for_question(cell, q)
    prompt = user_prompt(q["question"], evidence=text, label=LABEL, tool=False)
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
            prompt_eval_count=r["prompt_eval_count"],
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
            prompt_eval_count=0,
            duration=0.0,
            done_reason="error",
            error=repr(exc),
        )
    return row


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument(
        "--retry-only", action="store_true", help="redo failed calls; no probes, no meta"
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
        logger.info("copied 17 rows", extra={"ctx": {"n": len(old)}})
        have = read_raw(raw)
    embed, cache = embedder()
    ev = Evidence(questions, cells=cells, embed=embed, cache=cache)
    probes = []
    first17 = {(r["model"], r["condition"], r["qid"]): r for r in have if r.get("reused_from")}

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

    redo: set[tuple[str, str]] = set()
    for m, c in [] if args.retry_only else sorted(REUSED):  # reused first: a mismatch shows early
        if m not in models or c not in cells:
            continue
        n0 = len(probes)
        for q in spread(questions, PROBES_REUSED):
            probe(m, c, q, first17.get((m, c, q["id"])))
        same = sum(p["same"] for p in probes[n0:])
        logger.info("reused probes", extra={"ctx": {"model": m, "condition": c, "same": same}})
        if same < len(probes) - n0:
            logger.warning("reused cell drifted, asking it again", extra={"ctx": {"model": m}})
            redo.add((m, c))
    done = {k for k in done_keys(have) if k[:2] not in redo}
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
    meta17 = json.loads(META17.read_text("utf-8"))
    meta = {
        "date": date.today().isoformat(),
        "machine": platform.node(),
        "ollama": ollama.version(),
        "models": digests,
        "models_17": meta17["models"],
        "same_digest_as_17": True,  # checked at start (check_digests)
        "ollama_17": meta17.get("ollama"),
        "think": {m: THINK[m] for m in models},
        "options": OPTIONS,
        "budgets": {c: BUDGETS[c] for c in cells},
        "cells": cells,
        "reused": sorted(REUSED),
        "reused_redone": sorted(redo),
        "n_questions": len(questions),
        "drift": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_cell.items())},
        "drift_rows": probes,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
