"""Reading track: the lookup tool on C1/C4/C5 (96 questions) against the host ollama
-> results/read/raw.jsonl (resumable). Step 1: question + table catalog -> the cells the model
wants (LOOKUP_SCHEMA); the program resolves them (lookup.py); step 2: 17's user_prompt with the
cells as evidence. gpt-oss runs the lookup cell twice (lookup, lookup-2): it does not repeat
itself (17, 18), so the second run is the noise floor. The baselines are copied, not asked:
cards from 18 (qwen3's from 17, gpt-oss's 18 rerun) and oracle from 17 (16's), each checked by
re-asking 2 questions with the stored evidence (qwen3 must be byte-identical, else stop).

    uv run python experiments/19-lookup-tool/run_read.py [--models a,b] [--limit N] [--retry-only]
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
sys.path.insert(0, str(HERE))
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
RAW18 = HERE.parent / "18-table-kv/results/read/raw.jsonl"
META18 = HERE.parent / "18-table-kv/results/read/meta.json"
RAW17 = HERE.parent / "17-table-retrieval/results/read/raw.jsonl.gz"
CHAINS = ("C1", "C4", "C5")
MODELS = ["qwen3:8b", "gpt-oss:20b"]
THINK: dict[str, bool | str] = {"qwen3:8b": False, "gpt-oss:20b": "low"}  # see 02 README
OPTIONS = {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 1024}
RUNS = {  # first run (lookup*: 2026-11-01 01:56) and the fix run (fix*: example-free prompt,
    # brackets ignored in names), agreed with Mike after the first run
    "qwen3:8b": ["lookup", "fix"],
    "gpt-oss:20b": ["lookup", "lookup-2", "fix", "fix-2"],
}
REUSED = {"cards": "18", "oracle": "17"}  # cell -> experiment it is copied from
LABELS = {"cards": "檢索到的段落，可能來自不同條文與附表", "oracle": "規定與附表節錄"}  # 17's
PROBES = 2


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


def reused_rows(qids: set[str], models: list[str]) -> list[dict]:
    """18's cards and 17's oracle rows for these questions, the last good row per key (what their
    score.py used), marked reused_from."""
    with gzip.open(RAW17, "rt", encoding="utf-8") as f:
        r17 = _last_good([json.loads(ln) for ln in f if ln.strip()])
    r18 = _last_good(read_raw(RAW18))
    out = []
    for cell, src in REUSED.items():
        rows = r18 if src == "18" else r17
        for (m, c, qid), r in sorted(rows.items()):
            if c == cell and m in models and qid in qids:
                out.append({**r, "reused_from": src})
    return out


def check_digests(models: list[str]) -> dict[str, str]:
    """The chat models must be the ones 18 ran (17's are the same: 18's meta)."""
    meta18 = json.loads(META18.read_text("utf-8"))
    digests = {m: ollama.model_digest(m) for m in models}
    bad = {m: (d, meta18["models"].get(m)) for m, d in digests.items() if d != meta18["models"][m]}
    if bad:
        raise SystemExit(f"model digests differ from 18: {bad}")
    return digests


def spread(questions: list[dict], n: int) -> list[dict]:
    step = max(1, len(questions) // n)
    return questions[::step][:n]


def _chat(prompt: str, model: str, schema: dict) -> dict:
    return ollama.chat(
        prompt, model=model, system=SYSTEM, format=schema, think=THINK[model], options=OPTIONS
    )


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
    row: dict = {"model": model, "condition": cell, "qid": q["id"], "chain": q["chain"]}
    fix = cell.startswith("fix")
    p1 = lookup_prompt(q["question"], catalog=cat, fix=fix)
    try:
        r1 = _chat(p1, model, LOOKUP_SCHEMA)
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
            duration=r1["duration"] + r2["duration"],
            done_reason=r2["done_reason"],
            error="",
        )
    except (OSError, KeyError, ValueError) as exc:
        logger.exception(
            "call failed", extra={"ctx": {"model": model, "condition": cell, "qid": q["id"]}}
        )
        row.update(content="", raw_content="", duration=0.0, done_reason="error", error=repr(exc))
    return row


def reask(r: dict, q: dict) -> str:
    """A copied row asked again with its stored evidence and 17's prompt -> the raw reply."""
    prompt = user_prompt(
        q["question"], evidence=r["evidence_text"], label=LABELS[r["condition"]], tool=False
    )
    return _chat(prompt, r["model"], SCHEMA)["content"]


def same_answer(a: str, b: str) -> bool:
    """found and value equal (gpt-oss: its raw text varies between identical calls)."""
    try:
        x, y = json.loads(a), json.loads(b)
    except json.JSONDecodeError:
        return a == b
    return (x.get("found"), x.get("value")) == (y.get("found"), y.get("value"))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument(
        "--retry-only", action="store_true", help="redo failed calls; no probes, no meta"
    )
    args = ap.parse_args(argv)
    models = args.models.split(",")
    setup_logging("rag_lab")
    questions = load_questions()
    if args.limit:
        questions = questions[: args.limit]
    byid = {q["id"]: q for q in questions}
    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.out / "raw.jsonl"
    digests = check_digests(models)
    have = read_raw(raw)
    copied = {(r["model"], r["condition"], r["qid"]) for r in have if r.get("reused_from")}
    old = [
        r
        for r in reused_rows(set(byid), models)
        if (r["model"], r["condition"], r["qid"]) not in copied
    ]
    if old:
        with raw.open("a", encoding="utf-8") as f:
            for r in old:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("copied rows", extra={"ctx": {"n": len(old)}})
        have = read_raw(raw)
    probes = []
    if not args.retry_only:  # copied cells first: a mismatch shows before the long part
        reused = {k: r for k, r in _last_good(have).items() if r.get("reused_from")}
        for m in models:
            for cell in REUSED:
                for q in spread(questions, PROBES):
                    ref = reused[(m, cell, q["id"])]
                    again = reask(ref, q)
                    probes.append(
                        {
                            "model": m,
                            "condition": cell,
                            "qid": q["id"],
                            "same": again == ref["raw_content"],
                            "same_answer": same_answer(again, ref["raw_content"]),
                            "content": again,
                            "first": ref["raw_content"],
                        }
                    )
        logger.info("reused probes", extra={"ctx": {"probes": [p["same"] for p in probes]}})
        if not all(p["same"] for p in probes if p["model"] == "qwen3:8b"):
            raise SystemExit("qwen3 reused rows drifted: look before asking anything else")
    cat = catalog()
    done = set(_last_good(have))
    todo = [k for k in grid(models, questions) if k not in done]
    logger.info("grid", extra={"ctx": {"todo": len(todo), "catalog_chars": len(cat)}})
    secs: dict[tuple[str, str], list[float]] = {}
    with raw.open("a", encoding="utf-8") as f:
        for i, (m, c, qid) in enumerate(todo, 1):
            row = ask_lookup(m, c, byid[qid], cat)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            s = secs.setdefault((m, c), [])
            s.append(row["duration"])
            if len(s) == 3:
                logger.info(
                    "cell timing",
                    extra={"ctx": {"model": m, "condition": c, "sec_per_q": round(sum(s) / 3, 1)}},
                )
            if i % 20 == 0 or i == len(todo):
                logger.info("progress", extra={"ctx": {"done": i, "todo": len(todo)}})
    if args.retry_only:
        return 0
    first = _last_good(read_raw(raw))
    asked_now = sorted({(m, c) for m, c, _ in todo})
    for m, c in asked_now:  # probes for the cells asked in this invocation only
        for q in spread(questions, PROBES):
            again = ask_lookup(m, c, q, cat)
            ref = first.get((m, c, q["id"]))
            probes.append(
                {
                    "model": m,
                    "condition": c,
                    "qid": q["id"],
                    "same": ref is not None and again["raw_content"] == ref["raw_content"],
                    "same_answer": ref is not None
                    and same_answer(again["raw_content"], ref["raw_content"]),
                    "content": again["raw_content"],
                    "first": ref["raw_content"] if ref else None,
                }
            )
    by_cell: dict[str, list[int]] = {}
    for p in probes:
        by_cell.setdefault(f"{p['model']} {p['condition']}", []).append(int(p["same"]))
    drift = {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_cell.items())}
    logger.info("drift", extra={"ctx": drift})
    meta = {
        "date": date.today().isoformat(),
        "machine": platform.node(),
        "ollama": ollama.version(),
        "models": digests,
        "same_digest_as_18": True,  # checked at start (check_digests)
        "think": {m: THINK[m] for m in models},
        "options": OPTIONS,
        "runs": {m: RUNS[m] for m in models},
        "fix": "fix cells: prompts._LOOKUP (no examples), lookup.resolve(strict=False)",
        "reused": REUSED,
        "n_questions": len(questions),
        "catalog_chars": len(cat),
        "drift": drift,
        "drift_rows": probes,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
