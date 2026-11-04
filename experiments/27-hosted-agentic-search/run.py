"""Claude answers 16's 160 rule-lookup questions with a search tool -> results/raw.jsonl.

    uv run python experiments/27-hosted-agentic-search/run.py [--sample N] [--budget 25]

One condition, `agent`: 17's prompt with the evidence line replaced by the search tool (prompts.py),
up to 5 searches (claude.py). The tool is 17's cards retrieval for whatever query the model writes
(search.py). Haiku answers twice (agent-2 is the noise floor), Opus once. The baseline is 26's rows
(cards, decomp@4000, oracle: the same models, the same day, the same ruler), read by score.py.
Per question it keeps the queries, the turns, the union of the labels the searches returned (17's
hit fields) and every token billed, cache writes and reads apart. Every turn is billed, failed
and stopped questions too; the run stops once the cost passes --budget. A stop (refusal,
max_tokens, the turn cap) is the model's answer (kept, scored wrong); a failed call asks the whole
question again on the next run; a final row counts as done only while its signature (system,
prompt, tool, caps) is the one the inputs give now.
"""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(HERE))
import claude  # noqa: E402
import search  # noqa: E402
from evidence import DEPTH, hit_stats  # noqa: E402
from prompts import SCHEMA, SYSTEM, TOOL, agent_prompt  # noqa: E402

logger = logging.getLogger(__name__)

HAIKU, OPUS = "claude-haiku-5-5", "claude-opus-5-5"
MODELS = [HAIKU, OPUS]  # the cheap one first
RUNS = {HAIKU: ["agent", "agent-2"], OPUS: ["agent"]}
CONDITIONS = RUNS[HAIKU]  # every condition name that can appear
HIT_FIELDS = ("rows_hit", "rows_total", "tables_hit", "tables_total", "article_hit")
TOKENS = {  # usage key -> row field
    "input": "input_tokens",
    "cache_write": "cache_write_tokens",
    "cache_read": "cache_read_tokens",
    "output": "output_tokens",
    "calls": "api_calls",
}
BUDGET = 25.0  # US$; the plan says stop and ask Mike above it
MAX_FAILS_IN_A_ROW = 5  # a repeating error (a 400 on every question) stops the run
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
    """Keys with a final row (an answer or a stop) asked with the signature the inputs give now."""
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


def prompt_for(q: dict) -> str:
    return agent_prompt(q["question"])


def signature(q: dict) -> str:
    """sha256 of everything that shapes the question: system, prompt, tool, the caps, effort and
    max_tokens, and the search's budget, depth and embedding model."""
    sent = [
        SYSTEM,
        prompt_for(q),
        TOOL,
        claude.MAX_SEARCHES,
        claude.MAX_TURNS,
        claude.LIMIT_MSG,
        claude.EFFORT,
        claude.MAX_TOKENS,
        search.BUDGET,
        DEPTH,
        search.BGE_DIGEST,
    ]
    return hashlib.sha256(json.dumps(sent, ensure_ascii=False).encode("utf-8")).hexdigest()


def signatures(questions: list[dict]) -> dict[tuple[str, str], str]:
    return {(c, q["id"]): signature(q) for c in CONDITIONS for q in questions}


def make_searcher():
    return search.Searcher(*search.embedder())


def usage_of(row: dict) -> dict[str, int]:
    return {k: row.get(f, 0) for k, f in TOKENS.items()}


def ask(model: str, cond: str, q: dict, searcher) -> dict:
    prompt = prompt_for(q)
    queries: list[str] = []
    labels: set[str] = set()
    chars = [0]

    def run_tool(query: str) -> str:
        text, labs = searcher.search(query)
        queries.append(query)
        labels.update(labs)
        chars[0] += len(text)
        return text

    usage = claude.zero_usage()
    row: dict = {
        "model": model,
        "condition": cond,
        "qid": q["id"],
        "chain": q["chain"],
        "run": 2 if cond.endswith("-2") else 1,
        "date": date.today().isoformat(),
        "prompt_len": len(prompt),
        "prompt_sha": signature(q),
    }
    try:
        r = claude.agent(
            prompt,
            model=model,
            system=SYSTEM,
            schema=SCHEMA,
            tool=TOOL,
            run_tool=run_tool,
            usage=usage,
        )
        row.update(
            content=r["content"],
            turns=r["turns"],
            refused_searches=r["refused"],
            served_model=r.get("served_model", ""),
            request_ids=r.get("request_ids", []),
            duration=r["duration"],
            done_reason=r["done_reason"],
            error="",
        )
    except claude.StopError as exc:  # the model's answer: final, scored wrong, never re-asked
        logger.warning(
            "question stopped",
            extra={
                "ctx": {"model": model, "condition": cond, "qid": q["id"], "reason": exc.reason}
            },
        )
        row.update(
            content="",
            turns=exc.turns,
            refused_searches=exc.refused,
            served_model=exc.served_model,
            request_ids=exc.request_ids,
            duration=0.0,
            done_reason="stopped",
            stopped=exc.reason or "unknown",
            error="",
        )
    except CALL_ERRORS as exc:  # network or server: the whole question is asked again next run
        logger.exception(
            "call failed", extra={"ctx": {"model": model, "condition": cond, "qid": q["id"]}}
        )
        row.update(content="", duration=0.0, done_reason="error", error=repr(exc))
    hits = hit_stats(labels, q)
    row.update(
        queries=queries,
        n_searches=len(queries),
        evidence_chars=chars[0],
        **{k: hits[k] for k in HIT_FIELDS},
        **{f: usage[k] for k, f in TOKENS.items()},
    )
    return row


def spent_usd(rows: list[dict]) -> float:
    """Every row ever written, failed and superseded ones too: they were billed."""
    return sum(claude.price(r["model"], usage_of(r)) for r in rows)


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
    searcher = make_searcher()
    cache = getattr(searcher, "cache", None)
    fails = 0
    with raw.open("a", encoding="utf-8") as f:
        for i, (m, c, qid) in enumerate(todo, 1):
            row = ask(m, c, byid[qid], searcher)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            spent += claude.price(m, usage_of(row))
            fails = fails + 1 if row["error"] else 0
            if fails >= MAX_FAILS_IN_A_ROW:
                logger.error("failing in a row, stopping", extra={"ctx": {"fails": fails}})
                return 3
            if i % 20 == 0 or i == len(todo):
                if cache is not None:
                    cache.save()  # the models' queries, embedded once
                logger.info(
                    "progress",
                    extra={"ctx": {"done": i, "todo": len(todo), "spent_usd": round(spent, 4)}},
                )
            if spent > args.budget:
                if cache is not None:
                    cache.save()
                logger.error("budget passed, stopping", extra={"ctx": {"spent_usd": spent}})
                return 2
    every = read_raw(raw)  # every question ever asked, failed and superseded ones too: all billed
    rows = last_rows(raw)
    good = [r for r in rows if not r.get("error")]
    meta = {
        "date": date.today().isoformat(),
        "machine": platform.node(),
        "anthropic": version("anthropic"),
        "models": MODELS,
        "runs": RUNS,
        "served_models": sorted({r["served_model"] for r in good if r.get("served_model")}),
        "effort": claude.EFFORT,
        "max_tokens_per_turn": claude.MAX_TOKENS,
        "max_searches": claude.MAX_SEARCHES,
        "max_turns": claude.MAX_TURNS,
        "n_questions": len(questions),
        "tokens": {
            m: {f: sum(r.get(f, 0) for r in every if r["model"] == m) for f in TOKENS.values()}
            for m in MODELS
        },
        "questions_asked": len(every),
        "api_calls": sum(r.get("api_calls", 0) for r in every),
        "failed_questions": sum(bool(r.get("error")) for r in every),
        "failed_left": sum(bool(r.get("error")) for r in rows),
        "stopped": sum(bool(r.get("stopped")) for r in rows),
        "spent_usd": round(spent_usd(every), 4),
        "prices_usd_per_mtok": {m: claude.PRICES[m] for m in MODELS},
        "cache_multipliers": {"write": claude.CACHE_WRITE, "read": claude.CACHE_READ},
    }
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
