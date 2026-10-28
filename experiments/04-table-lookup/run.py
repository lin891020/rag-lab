"""Run the grid (models x conditions x questions) against the host ollama -> results/raw.jsonl.

    uv run python experiments/04-table-lookup/run.py [--models qwen3:8b,gpt-oss:20b]

Resumable. Tables come from corpus/tables.json (no PDFs needed). Conditions (prompts.py):
md (03's markdown), md+range (bound columns), md+rows (top-k rows by the selector),
md+range+rows (both); the selector always runs on the un-expanded table so the 2x2 is clean.
Each row records which rows were kept and whether the gold row was among them (row_hit).
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from collections.abc import Iterator
from datetime import date
from importlib.metadata import version as pkg_version
from pathlib import Path

from rag_lab import ollama
from rag_lab.log import setup_logging

HERE = Path(__file__).parent
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE))
from prompts import CONDITIONS, SCHEMA, SYSTEM, TOP_K, user_prompt  # noqa: E402
from select_rows import select_rows, subtable_markdown  # noqa: E402
from tables import Table, expand_ranges, table_from_json, to_markdown  # noqa: E402

logger = logging.getLogger(__name__)

MODELS = ["qwen3:8b", "gpt-oss:20b"]
THINK: dict[str, bool | str] = {"qwen3:8b": False, "gpt-oss:20b": "low"}
OPTIONS = {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 1024}
DRIFT_PER_CELL = 2


def grid(
    models: list[str], conditions: list[str], questions: list[dict]
) -> Iterator[tuple[str, str, str]]:
    for m in models:
        for c in conditions:
            for q in questions:
                yield (m, c, q["id"])


def done_keys(raw: Path) -> set[tuple[str, str, str]]:
    if not raw.exists():
        return set()
    out = set()
    for ln in raw.read_text("utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            out.add((r["model"], r["condition"], r["qid"]))
    return out


def drift_sample(
    models: list[str], conditions: list[str], questions: list[dict], *, per_cell: int
) -> list[tuple[str, str, str]]:
    ids = [q["id"] for q in questions[:per_cell]]
    return [(m, c, q) for m in models for c in conditions for q in ids]


def load_questions() -> list[dict]:
    return [
        json.loads(ln)
        for ln in (HERE / "questions.jsonl").read_text("utf-8").splitlines()
        if ln.strip()
    ]


def load_tables() -> dict[str, Table]:
    blob = json.loads((HERE / "corpus/tables.json").read_text("utf-8"))
    return {tid: table_from_json(d) for tid, d in blob.items()}


def evidence_for(q: dict, condition: str, tables: dict[str, Table]) -> tuple[str, str, dict]:
    """(evidence text, label, bookkeeping: rows kept + whether the gold row survived)."""
    show_range, keep_rows = CONDITIONS[condition]  # type: ignore[misc]
    parts, info = [], {"rows_kept": None, "row_hit": None}
    for g in q["gold"]:
        t = tables[g]
        shown = expand_ranges(t) if show_range else t
        if keep_rows:
            idxs = select_rows(t, q["question"], k=TOP_K)
            text = subtable_markdown(shown, idxs)
            key = q["row"]
            info = {
                "rows_kept": idxs,
                "row_hit": any(t.data_rows()[i][: len(key)] == key for i in idxs),
            }
        else:
            text = to_markdown(shown)
        parts.append(f"【{g}】\n{text}")
    return "\n\n".join(parts), "、".join(q["gold"]), info


def ask(model: str, condition: str, q: dict, tables: dict[str, Table]) -> dict:
    if CONDITIONS[condition] is None:
        text, label, info = None, "", {"rows_kept": None, "row_hit": None}
    else:
        text, label, info = evidence_for(q, condition, tables)
    prompt = user_prompt(q["question"], evidence=text, label=label)
    row = {
        "model": model,
        "condition": condition,
        "qid": q["id"],
        "prompt_len": len(prompt),
        **info,
    }
    try:
        r = ollama.chat(
            prompt, model=model, system=SYSTEM, format=SCHEMA, think=THINK[model], options=OPTIONS
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    args = ap.parse_args(argv)
    models = args.models.split(",")
    setup_logging("rag_lab")
    RESULTS.mkdir(exist_ok=True)
    raw = RESULTS / "raw.jsonl"
    questions = load_questions()
    tables = load_tables()
    conditions = list(CONDITIONS)
    byid = {q["id"]: q for q in questions}
    done = done_keys(raw)
    todo = [k for k in grid(models, conditions, questions) if k not in done]
    logger.info(
        "grid",
        extra={"ctx": {"total": len(models) * len(conditions) * len(questions), "todo": len(todo)}},
    )
    with raw.open("a", encoding="utf-8") as f:
        for i, (m, c, qid) in enumerate(todo, 1):
            f.write(json.dumps(ask(m, c, byid[qid], tables), ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(todo):
                logger.info(
                    "progress",
                    extra={"ctx": {"done": i, "todo": len(todo), "model": m, "condition": c}},
                )

    first = {
        (r["model"], r["condition"], r["qid"]): r["content"]
        for r in map(json.loads, raw.read_text("utf-8").splitlines())
        if r
    }
    probes = []
    for m, c, qid in drift_sample(models, conditions, questions, per_cell=DRIFT_PER_CELL):
        again = ask(m, c, byid[qid], tables)
        probes.append(
            {
                "model": m,
                "condition": c,
                "qid": qid,
                "same": again["content"] == first.get((m, c, qid)),
            }
        )
    same = sum(p["same"] for p in probes)
    logger.info("drift", extra={"ctx": {"same": same, "n": len(probes)}})

    # mean evidence length per condition (one model is enough: the prompt is model-independent)
    chars = {}
    for c in conditions:
        if CONDITIONS[c] is None:
            continue
        chars[c] = round(
            sum(len(evidence_for(q, c, tables)[0]) for q in questions) / len(questions)
        )
    meta = {
        "date": date.today().isoformat(),
        "machine": platform.node(),
        "ollama": ollama.version(),
        "models": {m: ollama.model_digest(m) for m in models},
        "think": {m: THINK[m] for m in models},
        "options": OPTIONS,
        "pdfplumber": pkg_version("pdfplumber"),
        "conditions": {c: v for c, v in CONDITIONS.items()},
        "top_k": TOP_K,
        "mean_evidence_chars": chars,
        "n_questions": {
            "total": len(questions),
            **{
                cat: sum(q["cat"] == cat for q in questions)
                for cat in ("range", "wide", "range+wide", "plain")
            },
        },
        "drift": {"same": same, "n": len(probes), "rows": probes},
    }
    (RESULTS / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
