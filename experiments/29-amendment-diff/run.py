"""Claude answers 29's 112 amendment questions in three conditions -> results/raw.jsonl.

    uv run python experiments/29-amendment-diff/run.py [--sample N | --per-kind N] [--kind T2] [--budget 20]

28's run.py reshaped: one tool (search_old over the 110 version; the question carries the new
article), and two conditions without tools. Conditions:
  agent    claude.agent with search_old (up to 5 searches, 8 turns, 27's loop); Haiku twice
  oracle   claude.chat with the old article appended; T2 and T3 only (T1 would be given away)
  closed   claude.chat, no evidence, one instruction (answer from memory); description only
Per question it keeps the queries, the union of labels the searches returned, whether the gold
old article was in that union (old_hit), and every token billed, cache writes and reads apart.
Every turn is billed, failed and stopped questions too; the run stops once the cost passes
--budget. A stop (refusal, max_tokens, the turn cap) is the model's answer (kept, scored wrong);
a failed call asks the whole question again on the next run; a final row counts as done only
while its signature (system, prompt, tool, caps, index) is the one the inputs give now.
"""

from __future__ import annotations

import argparse
import functools
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
from prompts import SCHEMAS, SYSTEM, TOOL, prompt_for  # noqa: E402

logger = logging.getLogger(__name__)

HAIKU, OPUS = "claude-haiku-5-5", "claude-opus-5-5"
MODELS = [HAIKU, OPUS]  # the cheap one first
RUNS = {HAIKU: ["closed", "agent", "agent-2", "oracle"], OPUS: ["closed", "agent", "oracle"]}
CONDITIONS = RUNS[HAIKU]  # every condition name that can appear
ORACLE_KINDS = ("T2", "T3")  # T1's oracle would hand over the answer
KINDS = ("T1", "T2", "T3")
TOKENS = {  # usage key -> row field
    "input": "input_tokens",
    "cache_write": "cache_write_tokens",
    "cache_read": "cache_read_tokens",
    "output": "output_tokens",
    "calls": "api_calls",
}
BUDGET = 20.0  # US$; the plan says stop and ask Mike above it
MAX_FAILS_IN_A_ROW = 5  # a repeating error (a 400 on every question) stops the run
CALL_ERRORS = (OSError, ValueError, anthropic.APIError)  # a KeyError is a bug: let it crash


def base_cond(cond: str) -> str:
    return cond[:-2] if cond.endswith("-2") else cond


def asks(cond: str, q: dict) -> bool:
    """Is this question asked in this condition? oracle skips T1."""
    return base_cond(cond) != "oracle" or q["kind"] in ORACLE_KINDS


def grid(questions: list[dict], models: list[str] = MODELS) -> Iterator[tuple[str, str, str]]:
    for m in models:
        for c in RUNS[m]:
            for q in questions:
                if asks(c, q):
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


def per_kind(questions: list[dict], n: int) -> list[dict]:
    """The first n of each kind, spread over the kind's list (trial runs)."""
    out = []
    for k in KINDS:
        out += sample([q for q in questions if q["kind"] == k], n)
    return out


@functools.cache
def old_texts() -> dict[str, str]:
    """Old article number -> body (oracle's evidence)."""
    return {a.num: a.body for a in search.old_articles()}


def prompt(q: dict, cond: str) -> str:
    c = base_cond(cond)
    return prompt_for(q, c, old_texts()[q["old_num"]] if c == "oracle" else None)


def signature(q: dict, cond: str) -> str:
    """sha256 of everything that shapes the question: system, prompt (rules, schema, evidence
    line, the new article), the tool, the caps, effort and max_tokens, and the search's index,
    budget, depth and embedding model (agent only)."""
    c = base_cond(cond)
    sent: list = [c, SYSTEM, prompt(q, cond), claude.EFFORT, claude.MAX_TOKENS]
    if c == "agent":
        sent += [
            TOOL,
            claude.MAX_SEARCHES,
            claude.MAX_TURNS,
            claude.LIMIT_MSG,
            search.index_sha(),
            search.BUDGET,
            search.DEPTH,
            search.BGE_DIGEST,
        ]
    return hashlib.sha256(json.dumps(sent, ensure_ascii=False).encode("utf-8")).hexdigest()


def signatures(questions: list[dict]) -> dict[tuple[str, str], str]:
    return {(c, q["id"]): signature(q, c) for c in CONDITIONS for q in questions if asks(c, q)}


@functools.cache
def _embedder():
    return search.embedder()


def make_searcher():
    return search.Searcher(*_embedder())


def usage_of(row: dict) -> dict[str, int]:
    return {k: row.get(f, 0) for k, f in TOKENS.items()}


def ask(model: str, cond: str, q: dict, searcher) -> dict:
    c = base_cond(cond)
    p = prompt(q, cond)
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
        "kind": q["kind"],
        "chapter": q["chapter"],
        "run": 2 if cond.endswith("-2") else 1,
        "date": date.today().isoformat(),
        "prompt_len": len(p),
        "prompt_sha": signature(q, cond),
    }
    try:
        if c == "agent":
            r = claude.agent(
                p,
                model=model,
                system=SYSTEM,
                schema=SCHEMAS[q["kind"]],
                tool=TOOL,
                run_tool=run_tool,
                usage=usage,
            )
        else:
            r = claude.chat(p, model=model, system=SYSTEM, schema=SCHEMAS[q["kind"]], usage=usage)
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
    gold = set(q["gold_labels"])
    row.update(
        queries=queries,
        n_searches=len(queries),
        evidence_chars=chars[0],
        labels=sorted(labels),
        old_hit=(gold <= labels) if gold and c == "agent" else None,  # 新增 has no old article
        **{f: usage[k] for k, f in TOKENS.items()},
    )
    return row


def spent_usd(rows: list[dict]) -> float:
    """Every row ever written, failed and superseded ones too: they were billed."""
    return sum(claude.price(r["model"], usage_of(r)) for r in rows)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="n questions spread over the list")
    ap.add_argument("--per-kind", type=int, default=0, help="n questions of each kind (trials)")
    ap.add_argument("--kind", choices=KINDS, default=None, help="only this kind (trials)")
    ap.add_argument("--budget", type=float, default=BUDGET, help="stop once passed (US$)")
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args(argv)
    setup_logging("rag_lab")
    claude.load_env()
    questions = load_questions()
    if args.kind:
        questions = [q for q in questions if q["kind"] == args.kind]
    if args.per_kind:
        questions = per_kind(questions, args.per_kind)
    elif args.sample:
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
    searcher = None
    fails = 0
    with raw.open("a", encoding="utf-8") as f:
        for i, (m, c, qid) in enumerate(todo, 1):
            if base_cond(c) == "agent" and searcher is None:
                searcher = make_searcher()
            row = ask(m, c, byid[qid], searcher)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            spent += claude.price(m, usage_of(row))
            fails = fails + 1 if row["error"] else 0
            if fails >= MAX_FAILS_IN_A_ROW:
                logger.error("failing in a row, stopping", extra={"ctx": {"fails": fails}})
                return 3
            if i % 20 == 0 or i == len(todo):
                if searcher is not None:
                    searcher.cache.save()  # the models' queries, embedded once
                logger.info(
                    "progress",
                    extra={"ctx": {"done": i, "todo": len(todo), "spent_usd": round(spent, 4)}},
                )
            if spent > args.budget:
                if searcher is not None:
                    searcher.cache.save()
                logger.error("budget passed, stopping", extra={"ctx": {"spent_usd": spent}})
                return 2
    every = read_raw(raw)  # every question ever asked, failed and superseded ones too: all billed
    rows = last_rows(raw)
    good = [r for r in rows if not r.get("error")]
    manifest = HERE / "corpus/manifest.json"
    meta = {
        "date": date.today().isoformat(),
        "machine": platform.node(),
        "anthropic": version("anthropic"),
        "models": MODELS,
        "runs": RUNS,
        "oracle_kinds": list(ORACLE_KINDS),
        "served_models": sorted({r["served_model"] for r in good if r.get("served_model")}),
        "effort": claude.EFFORT,
        "max_tokens_per_turn": claude.MAX_TOKENS,
        "max_searches": claude.MAX_SEARCHES,
        "max_turns": claude.MAX_TURNS,
        "index_sha": search.index_sha(),
        "bge_digest": search.BGE_DIGEST,
        "corpus": json.loads(manifest.read_text("utf-8")) if manifest.exists() else {},
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
