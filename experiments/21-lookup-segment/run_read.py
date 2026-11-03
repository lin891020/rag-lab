"""Reading track: 19's fix cells with the tool's answer widened to whole columns, against the host
ollama -> results/read/raw.jsonl (resumable). Step 1 is not asked again: each seg row takes the
step-1 reply of the 19 fix row it pairs with (seg <- fix, seg-2 <- fix-2), resolves the same
lookups and shows render_segment instead of render; only step 2 is asked. So the pointing is
identical and the only change is what the tool shows. 19's fix rows are copied (reused_from "19"),
their hit fields recounted with 20's hit_stats (needed rows only). Before asking, qwen3's copied
step 2 is re-asked on 2 questions with its stored evidence and must come back byte-identical.

    uv run python experiments/21-lookup-segment/run_read.py [--models a,b] [--limit N] [--retry-only]
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
sys.path.insert(0, str(HERE))
import cells  # noqa: E402
from evidence import hit_stats  # noqa: E402
from lookup import render_segment, resolve  # noqa: E402
from prompts import LOOKUP_LABEL, SCHEMA, SYSTEM, user_prompt  # noqa: E402

logger = logging.getLogger(__name__)

RESULTS = HERE / "results/read"
QUESTIONS = HERE / "questions.jsonl"
RAW19 = HERE.parent / "19-lookup-tool/results/read/raw.jsonl"
META19 = HERE.parent / "19-lookup-tool/results/read/meta.json"
CHAINS = ("C1", "C4", "C5")
MODELS = ["qwen3:8b", "gpt-oss:20b"]
THINK: dict[str, bool | str] = {"qwen3:8b": False, "gpt-oss:20b": "low"}  # 19's
OPTIONS = {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 1024}  # 19's
RUNS = {"qwen3:8b": ["seg"], "gpt-oss:20b": ["seg", "seg-2"]}
SOURCE = {"seg": "fix", "seg-2": "fix-2"}  # whose step-1 reply a seg cell reuses
COPIED = {"qwen3:8b": ["fix"], "gpt-oss:20b": ["fix", "fix-2"]}
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


def step1_rows(qids: set[str]) -> dict[tuple[str, str, str], dict]:
    """19's fix rows (last good per key) for these questions: their step-1 replies are reused."""
    return {
        k: r
        for k, r in _last_good(read_raw(RAW19)).items()
        if k[1] in COPIED.get(k[0], []) and k[2] in qids
    }


def copied_rows(qids: set[str]) -> list[dict]:
    """19's fix rows, marked reused_from, hit fields recounted from their lookups (20's way)."""
    byid = {q["id"]: q for q in load_questions()}
    out = []
    for (_, _, qid), r in sorted(step1_rows(qids).items()):
        labels = {x["label"] for x in r["lookups"] if x["status"] in ("ok", "blank")}
        out.append({**r, "reused_from": "19", **hit_stats(labels, byid[qid])})
    return out


def check_digests(models: list[str]) -> dict[str, str]:
    """The chat models must be the ones 19 ran."""
    meta19 = json.loads(META19.read_text("utf-8"))
    digests = {m: ollama.model_digest(m) for m in models}
    bad = {m: (d, meta19["models"].get(m)) for m, d in digests.items() if d != meta19["models"][m]}
    if bad:
        raise SystemExit(f"model digests differ from 19: {bad}")
    return digests


def spread(questions: list[dict], n: int) -> list[dict]:
    step = max(1, len(questions) // n)
    return questions[::step][:n]


def _chat(prompt: str, model: str, schema: dict) -> dict:
    return ollama.chat(
        prompt, model=model, system=SYSTEM, format=schema, think=THINK[model], options=OPTIONS
    )


def honest_labels(labels: set[str], text: str, q: dict) -> set[str]:
    """A whole column labels every row; keep a gold row only if a gold value is shown on it, so
    the hit fields mean what they mean for fix (a gold row reached, in a gold column). An
    abstain question has no gold value: its labels are kept as shown."""
    if q["expect"]["kind"] == "abstain":
        return labels
    gold = set(cells.gold_labels(q))
    want = cells.gold_cells(q)
    hit = {
        lb
        for lb, _, v in cells.shown_cells(text)
        if lb in gold and any(cells.near(v, g) for g in want)
    }
    return (labels - gold) | hit


def ask_seg(model: str, cell: str, q: dict, src: dict) -> dict:
    """Step 2 only: src's lookups (19's step 1) shown as whole columns."""
    row: dict = {"model": model, "condition": cell, "qid": q["id"], "chain": q["chain"]}
    asked = [x["asked"] for x in src["lookups"]]
    results = [resolve(*a) for a in asked]  # 19's fix matching (strict=False), re-resolved
    text, labels = render_segment(results)
    labels = honest_labels(labels, text, q)
    p2 = user_prompt(q["question"], evidence=text, label=LOOKUP_LABEL, tool=False)
    row.update(
        step1_from=SOURCE[cell],
        step1_content=src["step1_content"],
        step1_problem=src.get("step1_problem", ""),
        lookups=results,
        prompt_len=len(p2),
        evidence_chars=len(text),
        evidence_text=text,
        **hit_stats(labels, q),
    )
    try:
        r2 = _chat(p2, model, SCHEMA)
        row.update(
            content=r2["content"],
            raw_content=r2["content"],
            expr_ok=None,
            thinking_len=len(r2["thinking"]),
            eval_count=r2["eval_count"],
            prompt_eval_count=r2["prompt_eval_count"],
            duration=r2["duration"],
            done_reason=r2["done_reason"],
            error="",
        )
    except (OSError, ValueError) as exc:
        logger.exception(
            "call failed", extra={"ctx": {"model": model, "condition": cell, "qid": q["id"]}}
        )
        row.update(content="", raw_content="", duration=0.0, done_reason="error", error=repr(exc))
    return row


def same_answer(a: str, b: str) -> bool:
    """found and value equal (gpt-oss: its raw text varies between identical calls)."""
    try:
        x, y = json.loads(a), json.loads(b)
    except json.JSONDecodeError:
        return a == b
    return (x.get("found"), x.get("value")) == (y.get("found"), y.get("value"))


def reask(r: dict, q: dict) -> str:
    """A copied fix row's step 2 asked again with its stored evidence -> the raw reply."""
    prompt = user_prompt(q["question"], evidence=r["evidence_text"], label=LOOKUP_LABEL, tool=False)
    return _chat(prompt, r["model"], SCHEMA)["content"]


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
        questions = spread(questions, args.limit)
    byid = {q["id"]: q for q in questions}
    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.out / "raw.jsonl"
    digests = check_digests(models)
    have = read_raw(raw)
    copied = {(r["model"], r["condition"], r["qid"]) for r in have if r.get("reused_from")}
    old = [
        r
        for r in copied_rows(set(byid))
        if r["model"] in models and (r["model"], r["condition"], r["qid"]) not in copied
    ]
    if old:
        with raw.open("a", encoding="utf-8") as f:
            for r in old:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("copied rows", extra={"ctx": {"n": len(old)}})
    src = step1_rows(set(byid))
    probes = []
    if not args.retry_only:  # copied cells first: a drift shows before the long part
        for m in models:
            for q in spread(questions, PROBES):
                ref = src[(m, "fix", q["id"])]
                again = reask(ref, q)
                probes.append(
                    {
                        "model": m,
                        "condition": "fix",
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
    done = set(_last_good(read_raw(raw)))
    todo = [k for k in grid(models, questions) if k not in done]
    logger.info("grid", extra={"ctx": {"todo": len(todo)}})
    with raw.open("a", encoding="utf-8") as f:
        for i, (m, c, qid) in enumerate(todo, 1):
            row = ask_seg(m, c, byid[qid], src[(m, SOURCE[c], qid)])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(todo):
                logger.info("progress", extra={"ctx": {"done": i, "todo": len(todo)}})
    if args.retry_only:
        return 0
    first = _last_good(read_raw(raw))
    for m, c in sorted({(m, c) for m, c, _ in todo}):  # probes for the cells asked now
        for q in spread(questions, PROBES):
            again = ask_seg(m, c, q, src[(m, SOURCE[c], q["id"])])
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
        "same_digest_as_19": True,  # checked at start (check_digests)
        "think": {m: THINK[m] for m in models},
        "options": OPTIONS,
        "runs": {m: RUNS[m] for m in models},
        "source": SOURCE,
        "copied": {m: COPIED[m] for m in models},
        "n_questions": len(questions),
        "drift": drift,
        "drift_rows": probes,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
