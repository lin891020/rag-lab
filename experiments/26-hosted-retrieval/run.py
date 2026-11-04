"""Claude answers 16's 160 rule-lookup questions with 17's evidence -> results/raw.jsonl.

    uv run python experiments/26-hosted-retrieval/run.py [--sample N] [--budget 25]

Conditions (17's prompt, word for word; the evidence is the text 17 stored for gpt-oss, which
both small models got byte for byte):
  closed       no evidence (the prompt's closed-book sentence)
  struct@1500  one article / one table row per chunk, dense top, 1,500 chars (16's L1-struct)
  cards        17's chosen retrieval: a card per table, ranked with the articles, ~3,000 chars
  decomp@4000  sub-questions retrieved one by one, ~4,000 chars (17's best)
  oracle       the gold article excerpt and the gold rows
Haiku answers every condition twice (-2 is the noise floor), Opus once (its noise floor is 20,
which asked cards and oracle twice with these prompts). Every call is billed into the running
cost, failed ones too, and the run stops once the cost passes --budget. A stop other than
end_turn is the model's answer (kept, scored wrong); a failed call is asked again on the next
run; a final row counts as done only while its prompt's sha256 is the one the inputs give now.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import platform
import sys
from collections.abc import Iterator
from datetime import date
from importlib.metadata import version
from pathlib import Path

import anthropic

from rag_lab.log import setup_logging

HERE = Path(__file__).parent
RESULTS = HERE / "results"
RAW17 = HERE.parent / "17-table-retrieval/results/read/raw.jsonl.gz"
sys.path.insert(0, str(HERE))
import claude  # noqa: E402
from prompts import SCHEMA, SYSTEM, user_prompt  # noqa: E402

logger = logging.getLogger(__name__)

HAIKU, OPUS = "claude-haiku-5-5", "claude-opus-5-5"
MODELS = [HAIKU, OPUS]  # the cheap one first
BASE = ["closed", "struct@1500", "cards", "decomp@4000", "oracle"]
RUNS = {HAIKU: BASE + [f"{c}-2" for c in BASE], OPUS: BASE}
CONDITIONS = RUNS[HAIKU]  # every condition name that can appear
SOURCE_MODEL = "gpt-oss:20b"  # whose stored evidence is reused (qwen3's is byte-identical)
LABELS = {"oracle": "規定與附表節錄", "retrieved": "檢索到的段落，可能來自不同條文與附表"}  # 17's
HIT_FIELDS = ("rows_hit", "rows_total", "tables_hit", "tables_total", "article_hit")
BUDGET = 25.0  # US$; the plan says stop and ask Mike above it
CALL_ERRORS = (OSError, ValueError, anthropic.APIError)  # a KeyError is a bug: let it crash


def grid(questions: list[dict], models: list[str] = MODELS) -> Iterator[tuple[str, str, str]]:
    for m in models:
        for c in RUNS[m]:
            for q in questions:
                yield (m, c, q["id"])


def read_raw(raw: Path) -> list[dict]:
    if not raw.exists():
        return []
    return [json.loads(ln) for ln in raw.read_text("utf-8").splitlines() if ln.strip()]


def last_rows(raw: Path) -> list[dict]:
    """One row per (model, condition, qid): the last final one, else the last failed one."""
    best: dict[tuple[str, str, str], dict] = {}
    for r in read_raw(raw):
        k = (r["model"], r["condition"], r["qid"])
        if k not in best or not r.get("error") or best[k].get("error"):
            best[k] = r
    return list(best.values())


def done_keys(raw: Path, sig: dict[tuple[str, str], str]) -> set[tuple[str, str, str]]:
    """Keys with a final row (an answer or a stop) asked with the prompt the inputs give now."""
    return {
        (r["model"], r["condition"], r["qid"])
        for r in last_rows(raw)
        if not r.get("error") and r.get("prompt_sha") == sig.get((r["condition"], r["qid"]))
    }


def load_questions() -> list[dict]:
    return [
        json.loads(ln)
        for ln in (HERE / "questions.jsonl").read_text("utf-8").splitlines()
        if ln.strip()
    ]


def sample(questions: list[dict], n: int) -> list[dict]:
    step = max(1, len(questions) // n)
    return questions[::step][:n]


def evidence17() -> dict[tuple[str, str], dict]:
    """(condition, qid) -> 17's last good gpt-oss row (evidence_text, prompt_len, hit fields)."""
    out: dict[tuple[str, str], dict] = {}
    with gzip.open(RAW17, "rt", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r["model"] == SOURCE_MODEL and r["condition"] in BASE and not r.get("error"):
                out[(r["condition"], r["qid"])] = r
    return out


def prompt_for(cond: str, q: dict, ev: dict[tuple[str, str], dict]) -> str:
    base = cond.removesuffix("-2")
    if base == "closed":
        return user_prompt(q["question"], evidence=None, label="", tool=False)
    label = LABELS["oracle"] if base == "oracle" else LABELS["retrieved"]
    text = ev[(base, q["id"])]["evidence_text"]
    return user_prompt(q["question"], evidence=text, label=label, tool=False)


def _text_sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def signatures(questions: list[dict]) -> dict[tuple[str, str], str]:
    """(condition, qid) -> sha256 of the prompt the inputs give now."""
    ev = evidence17()
    return {(c, q["id"]): _text_sha(prompt_for(c, q, ev)) for c in CONDITIONS for q in questions}


def ask(model: str, cond: str, q: dict, ev: dict[tuple[str, str], dict]) -> dict:
    base = cond.removesuffix("-2")
    prompt = prompt_for(cond, q, ev)
    src = ev.get((base, q["id"]), {})
    row: dict = {
        "model": model,
        "condition": cond,
        "qid": q["id"],
        "chain": q["chain"],
        "run": 2 if cond.endswith("-2") else 1,
        "date": date.today().isoformat(),
        "prompt_len": len(prompt),
        "prompt_sha": _text_sha(prompt),
        "evidence_chars": len(src.get("evidence_text", "")),
        **{k: src[k] for k in HIT_FIELDS if k in src},
    }
    try:
        r = claude.chat(prompt, model=model, system=SYSTEM, schema=SCHEMA)
        row.update(
            content=r["content"],
            input_tokens=r["prompt_eval_count"],
            output_tokens=r["eval_count"],
            prompt_eval_count=r["prompt_eval_count"],
            eval_count=r["eval_count"],
            served_model=r.get("served_model", ""),
            request_id=r.get("request_id", ""),
            duration=r["duration"],
            done_reason=r["done_reason"],
            error="",
        )
    except claude.StopError as exc:  # the model's answer: final, scored wrong, never re-asked
        logger.warning(
            "call stopped",
            extra={
                "ctx": {"model": model, "condition": cond, "qid": q["id"], "reason": exc.reason}
            },
        )
        row.update(
            content="",
            input_tokens=exc.tokens_in,
            output_tokens=exc.tokens_out,
            prompt_eval_count=exc.tokens_in,
            eval_count=exc.tokens_out,
            duration=0.0,
            done_reason="stopped",
            stopped=exc.reason or "unknown",
            error="",
        )
    except CALL_ERRORS as exc:  # network or server: asked again on the next run
        logger.exception(
            "call failed", extra={"ctx": {"model": model, "condition": cond, "qid": q["id"]}}
        )
        row.update(
            content="",
            input_tokens=0,
            output_tokens=0,
            duration=0.0,
            done_reason="error",
            error=repr(exc),
        )
    return row


def spent_usd(rows: list[dict]) -> float:
    """Every call ever written, failed and superseded ones too: they were billed."""
    return sum(
        claude.price(r["model"], r.get("input_tokens", 0), r.get("output_tokens", 0)) for r in rows
    )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="n questions spread over the list")
    ap.add_argument("--budget", type=float, default=BUDGET, help="stop once passed (US$)")
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args(argv)
    setup_logging("rag_lab")
    claude.load_env()
    questions = load_questions()
    if args.sample:
        questions = sample(questions, args.sample)
    ev = evidence17()
    missing = [(c, q["id"]) for c in BASE[1:] for q in questions if (c, q["id"]) not in ev]
    if missing:
        raise SystemExit(f"17's evidence missing: {missing[:5]}")
    sig = signatures(questions)
    byid = {q["id"]: q for q in questions}
    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.out / "raw.jsonl"
    spent = spent_usd(read_raw(raw))
    done = done_keys(raw, sig)
    todo = [k for k in grid(questions) if k not in done]
    logger.info("grid", extra={"ctx": {"todo": len(todo), "spent_usd": round(spent, 4)}})
    if spent > args.budget:
        logger.error("budget already passed", extra={"ctx": {"spent_usd": spent}})
        return 2
    with raw.open("a", encoding="utf-8") as f:
        for i, (m, c, qid) in enumerate(todo, 1):
            row = ask(m, c, byid[qid], ev)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            spent += claude.price(m, row["input_tokens"], row["output_tokens"])
            if i % 20 == 0 or i == len(todo):
                logger.info(
                    "progress",
                    extra={"ctx": {"done": i, "todo": len(todo), "spent_usd": round(spent, 4)}},
                )
            if spent > args.budget:
                logger.error("budget passed, stopping", extra={"ctx": {"spent_usd": spent}})
                return 2
    every = read_raw(raw)  # every call ever made, failed and superseded ones too: all billed
    rows = last_rows(raw)
    good = [r for r in rows if not r.get("error")]
    meta = {
        "date": date.today().isoformat(),
        "machine": platform.node(),
        "anthropic": version("anthropic"),
        "models": MODELS,
        "runs": RUNS,
        "served_models": sorted({r.get("served_model", "") for r in good}),
        "effort": claude.EFFORT,
        "max_tokens": claude.MAX_TOKENS,
        "evidence_from": f"17 {SOURCE_MODEL} rows",
        "n_questions": len(questions),
        "tokens": {
            m: [
                sum(r["input_tokens"] for r in every if r["model"] == m),
                sum(r["output_tokens"] for r in every if r["model"] == m),
            ]
            for m in MODELS
        },
        "calls": len(every),
        "failed_calls": sum(bool(r.get("error")) for r in every),
        "failed_left": sum(bool(r.get("error")) for r in rows),
        "stopped": sum(bool(r.get("stopped")) for r in rows),
        "spent_usd": round(spent_usd(every), 4),
        "prices_usd_per_mtok": {m: claude.PRICES[m] for m in MODELS},
    }
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
