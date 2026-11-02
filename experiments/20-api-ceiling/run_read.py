"""Reading track: Claude on 19's C1/C4/C5 questions (96) -> results/read/raw.jsonl (resumable).
Three cells, each asked twice (no temperature on the API, so the second run is the noise floor):
cards and oracle re-ask 17's user_prompt with the exact evidence 19 stored (gpt-oss's rows; both
small models had the same text); fix is 19's two-step lookup (example-free prompt, brackets
ignored). The small models are not asked: their 19 rows are copied (reused_from "19"). A running
cost is logged; the run stops when it passes --budget (US$).

    uv run python experiments/20-api-ceiling/run_read.py [--models a,b] [--limit N] [--budget 40]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
from collections.abc import Iterator
from datetime import date
from importlib.metadata import version
from pathlib import Path

import anthropic

from rag_lab.log import setup_logging

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import claude  # noqa: E402
from evidence import hit_stats  # noqa: E402
from lookup import catalog, render, resolve  # noqa: E402
from prompts import (  # noqa: E402
    LOOKUP_LABEL,
    LOOKUP_SCHEMA,
    MAX_LOOKUPS,
    SCHEMA,
    SYSTEM,
    lookup_prompt,
    user_prompt,
)

logger = logging.getLogger(__name__)

RESULTS = HERE / "results/read"
QUESTIONS = HERE / "questions.jsonl"
RAW19 = HERE.parent / "19-lookup-tool/results/read/raw.jsonl"
ENV = HERE.parents[1] / ".env"
CHAINS = ("C1", "C4", "C5")
MODELS = ["claude-opus-5-5", "claude-haiku-5-5"]
RUNS = {m: ["cards", "oracle", "fix", "cards-2", "oracle-2", "fix-2"] for m in MODELS}
COPIED = {
    "qwen3:8b": ["cards", "oracle", "fix"],
    "gpt-oss:20b": ["cards", "oracle", "fix", "fix-2"],
}
SOURCE_MODEL = "gpt-oss:20b"  # whose stored cards/oracle evidence is re-asked
LABELS = {"cards": "檢索到的段落，可能來自不同條文與附表", "oracle": "規定與附表節錄"}  # 17's
PRICES = {"claude-opus-5-5": (4.00, 20.00), "claude-haiku-5-5": (0.10, 0.50)}  # US$ / M tokens
HIT_FIELDS = ("rows_hit", "rows_total", "tables_hit", "tables_total", "article_hit")
CALL_ERRORS = (OSError, ValueError, anthropic.APIError)  # a KeyError is a bug: let it crash


def load_env(path: Path = ENV) -> None:
    """KEY=value lines into os.environ (existing variables win); the key is never logged."""
    if not path.exists():
        return
    for ln in path.read_text("utf-8").splitlines():
        k, sep, v = ln.strip().partition("=")
        if sep and k and not k.startswith("#"):
            os.environ.setdefault(k, v.strip().strip("'\""))


def price(model: str, tokens_in: int, tokens_out: int) -> float:
    pin, pout = PRICES[model]
    return round((tokens_in * pin + tokens_out * pout) / 1_000_000, 6)


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    qs = [json.loads(ln) for ln in path.read_text("utf-8").splitlines() if ln.strip()]
    return [q for q in qs if q["chain"] in CHAINS]


def grid(models: list[str], questions: list[dict]) -> Iterator[tuple[str, str, str]]:
    for m in models:
        for c in RUNS[m]:
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


def _last_good(rows: list[dict]) -> dict[tuple[str, str, str], dict]:
    last = {}
    for r in rows:
        if not r.get("error"):
            last[(r["model"], r["condition"], r["qid"])] = r
    return last


def sample(questions: list[dict], n: int) -> list[dict]:
    """n questions spread evenly over the list (so over the chains), for a cost trial."""
    step = max(1, len(questions) // n)
    return questions[::step][:n]


def _stop_tokens(exc: Exception) -> tuple[int, int]:
    return (exc.tokens_in, exc.tokens_out) if isinstance(exc, claude.StopError) else (0, 0)


def _rows19() -> dict[tuple[str, str, str], dict]:
    return _last_good(read_raw(RAW19))


def copied_rows(qids: set[str]) -> list[dict]:
    """19's small-model rows for the cells in COPIED (last good per key), marked reused_from.
    The fix rows' hit fields are recounted from their lookups with 20's hit_stats (needed rows
    only), so they are counted the same way as Claude's."""
    byid = {q["id"]: q for q in load_questions()}
    out = []
    for (m, c, qid), r in sorted(_rows19().items()):
        if c in COPIED.get(m, []) and qid in qids:
            row = {**r, "reused_from": "19"}
            if c.startswith("fix"):
                labels = {x["label"] for x in r["lookups"] if x["status"] in ("ok", "blank")}
                row.update(hit_stats(labels, byid[qid]))
            out.append(row)
    return out


def source_evidence(qids: set[str]) -> dict[tuple[str, str], dict]:
    """(cell, qid) -> the 19 row whose stored evidence cards / oracle re-ask."""
    return {
        (c, qid): r
        for (m, c, qid), r in _rows19().items()
        if m == SOURCE_MODEL and c in LABELS and qid in qids
    }


def _chat(prompt: str, model: str, schema: dict) -> dict:
    return claude.chat(prompt, model=model, system=SYSTEM, schema=schema)


def parse_lookups(content: str) -> tuple[list[dict], str]:
    """The step-1 reply -> (lookups, problem); at most MAX_LOOKUPS, missing fields as ''."""
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        return [], "unparseable"
    items = obj.get("lookups") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return [], "no lookups"
    items = [x for x in items if isinstance(x, dict)]
    problem = "over the limit" if len(items) > MAX_LOOKUPS else ""
    keys = ("table", "row", "column")
    return [{k: str(x.get(k, "")) for k in keys} for x in items[:MAX_LOOKUPS]], problem


def ask_lookup(model: str, cell: str, q: dict, cat: str) -> dict:
    row = _row(model, cell, q)
    fix = cell.startswith("fix")
    p1 = lookup_prompt(q["question"], catalog=cat, fix=fix)
    used = [0, 0]  # tokens billed so far, kept when step 2 fails
    try:
        r1 = _chat(p1, model, LOOKUP_SCHEMA)
        used = [r1["prompt_eval_count"], r1["eval_count"]]
        asked, problem = parse_lookups(r1["content"])
        results = [resolve(x["table"], x["row"], x["column"], strict=not fix) for x in asked]
        text = render(results)
        labels = {r["label"] for r in results if r["status"] in ("ok", "blank")}
        p2 = user_prompt(q["question"], evidence=text, label=LOOKUP_LABEL, tool=False)
        r2 = _chat(p2, model, SCHEMA)
        row.update(
            step1_content=r1["content"],
            step1_problem=problem,
            lookups=results,
            step1_prompt_len=len(p1),
            step1_prompt_eval_count=r1["prompt_eval_count"],
            step1_done_reason=r1["done_reason"],
            prompt_len=len(p2),
            evidence_chars=len(text),
            evidence_text=text,
            **hit_stats(labels, q),
            content=r2["content"],
            raw_content=r2["content"],
            expr_ok=None,
            thinking_len=len(r1["thinking"]) + len(r2["thinking"]),
            eval_count=r1["eval_count"] + r2["eval_count"],
            prompt_eval_count=r2["prompt_eval_count"],
            input_tokens=r1["prompt_eval_count"] + r2["prompt_eval_count"],
            output_tokens=r1["eval_count"] + r2["eval_count"],
            served_model=r2.get("served_model", ""),
            duration=r1["duration"] + r2["duration"],
            done_reason=r2["done_reason"],
            error="",
        )
    except CALL_ERRORS as exc:
        _failed(row, exc, used)
    return row


def _row(model: str, cell: str, q: dict) -> dict:
    return {
        "model": model,
        "condition": cell,
        "qid": q["id"],
        "chain": q["chain"],
        "date": date.today().isoformat(),
    }


def _failed(row: dict, exc: Exception, used: list[int]) -> None:
    logger.exception(
        "call failed",
        extra={"ctx": {"model": row["model"], "condition": row["condition"], "qid": row["qid"]}},
    )
    tin, tout = _stop_tokens(exc)
    row.update(
        content="",
        raw_content="",
        duration=0.0,
        done_reason="error",
        error=repr(exc),
        input_tokens=used[0] + tin,
        output_tokens=used[1] + tout,
    )


def given_prompt(cell: str, q: dict, src: dict) -> str:
    base = cell.removesuffix("-2")
    return user_prompt(q["question"], evidence=src["evidence_text"], label=LABELS[base], tool=False)


def ask_given(model: str, cell: str, q: dict, src: dict) -> dict:
    """cards / oracle: 17's prompt with the evidence 19 stored; the hit fields are 19's (its
    oracle rows, from 16/17, have no table fields)."""
    row = _row(model, cell, q)
    prompt = given_prompt(cell, q, src)
    row.update(
        prompt_len=len(prompt),
        evidence_chars=len(src["evidence_text"]),
        evidence_text=src["evidence_text"],
        **{k: src[k] for k in HIT_FIELDS if k in src},
        expr_ok=None,
        thinking_len=0,
    )
    try:
        r = _chat(prompt, model, SCHEMA)
        row.update(
            content=r["content"],
            raw_content=r["content"],
            eval_count=r["eval_count"],
            prompt_eval_count=r["prompt_eval_count"],
            input_tokens=r["prompt_eval_count"],
            output_tokens=r["eval_count"],
            served_model=r.get("served_model", ""),
            duration=r["duration"],
            done_reason=r["done_reason"],
            error="",
        )
    except CALL_ERRORS as exc:
        _failed(row, exc, [0, 0])
    return row


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0, help="n questions spread over the chains")
    ap.add_argument("--budget", type=float, default=40.0, help="stop when the run passes it (US$)")
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args(argv)
    models = args.models.split(",")
    setup_logging("rag_lab")
    load_env()
    questions = load_questions()
    if args.limit:
        questions = questions[: args.limit]
    if args.sample:
        questions = sample(questions, args.sample)
    byid = {q["id"]: q for q in questions}
    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.out / "raw.jsonl"
    have = read_raw(raw)
    copied = {(r["model"], r["condition"], r["qid"]) for r in have if r.get("reused_from")}
    old = [
        r for r in copied_rows(set(byid)) if (r["model"], r["condition"], r["qid"]) not in copied
    ]
    if old:
        with raw.open("a", encoding="utf-8") as f:
            for r in old:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("copied rows", extra={"ctx": {"n": len(old)}})
        have = read_raw(raw)
    spent = sum(
        price(r["model"], r.get("input_tokens", 0), r.get("output_tokens", 0))
        for r in have
        if r["model"] in PRICES
    )  # failed calls included: they were billed too
    src = source_evidence(set(byid))
    cat = catalog()
    done = set(_last_good(have))
    todo = [k for k in grid(models, questions) if k not in done]
    logger.info("grid", extra={"ctx": {"todo": len(todo), "spent_usd": round(spent, 2)}})
    with raw.open("a", encoding="utf-8") as f:
        for i, (m, c, qid) in enumerate(todo, 1):
            q = byid[qid]
            base = c.removesuffix("-2")
            if base == "fix":
                row = ask_lookup(m, c, q, cat)
            else:
                row = ask_given(m, c, q, src[(base, qid)])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            spent += price(m, row.get("input_tokens", 0), row.get("output_tokens", 0))
            if i % 20 == 0 or i == len(todo):
                logger.info(
                    "progress",
                    extra={"ctx": {"done": i, "todo": len(todo), "spent_usd": round(spent, 2)}},
                )
            if spent > args.budget:
                logger.error("budget passed, stopping", extra={"ctx": {"spent_usd": spent}})
                return 2
    allrows = [r for r in read_raw(raw) if r["model"] in PRICES]
    rows = [r for r in allrows if not r.get("error")]
    meta = {
        "date": date.today().isoformat(),
        "machine": platform.node(),
        "anthropic": version("anthropic"),
        "models": models,
        "served_models": sorted({r.get("served_model", "") for r in rows}),
        "effort": claude.EFFORT,
        "max_tokens": claude.MAX_TOKENS,
        "runs": {m: RUNS[m] for m in models},
        "copied": COPIED,
        "source_model": SOURCE_MODEL,
        "fix": "fix cells: prompts._LOOKUP (no examples), lookup.resolve(strict=False)",
        "n_questions": len(questions),
        "catalog_chars": len(cat),
        "tokens": {
            m: [
                sum(r.get("input_tokens", 0) for r in rows if r["model"] == m),
                sum(r.get("output_tokens", 0) for r in rows if r["model"] == m),
            ]
            for m in models
        },
        "dates": sorted({r["date"] for r in allrows}),
        "failed_calls": sum(bool(r.get("error")) for r in allrows),
        "spent_usd": round(spent, 2),
        "prices_usd_per_mtok": {m: PRICES[m] for m in models},
    }
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
