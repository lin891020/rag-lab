"""Run the grid (models x renderings x questions) against the host ollama -> results/raw.jsonl.

    uv run python experiments/24-ocr-header/run.py [--models qwen3:8b,gpt-oss:20b]

Resumable. Evidence: corpus/<route>/<表號>.txt (clean = 04's markdown from pdfplumber; glm2 and
glm2-flat built by build_corpus.py from 23's saved glm-ocr replies). Prompt, schema and model
settings are 03's.
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
from build_corpus import RAW23, build, sources  # noqa: E402
from prompts import CONDITIONS, SCHEMA, SYSTEM, user_prompt  # noqa: E402

logger = logging.getLogger(__name__)

MODELS = ["qwen3:8b", "gpt-oss:20b"]
THINK: dict[str, bool | str] = {"qwen3:8b": False, "gpt-oss:20b": "low"}  # see 02 README
OPTIONS = {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 1024}
DRIFT_PER_CELL = 2


def grid(
    models: list[str], conditions: list[str], questions: list[dict]
) -> Iterator[tuple[str, str, str]]:
    for m in models:
        for c in conditions:
            for q in questions:
                yield (m, c, q["id"])


def last_rows(raw: Path) -> list[dict]:
    """One row per (model, condition, qid): the last good one, else the last failed one."""
    if not raw.exists():
        return []
    best: dict[tuple[str, str, str], dict] = {}
    for ln in raw.read_text("utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            k = (r["model"], r["condition"], r["qid"])
            if k not in best or not r.get("error") or best[k].get("error"):
                best[k] = r
    return list(best.values())


def done_keys(
    raw: Path, lens: dict[tuple[str, str], int] | None = None
) -> set[tuple[str, str, str]]:
    """Keys with a good row; a failed call (timeout while a model reloads) is asked again, and so
    is a row whose prompt is not the one the corpus builds now (lens: (condition, qid) -> length)."""
    return {
        (r["model"], r["condition"], r["qid"])
        for r in last_rows(raw)
        if not r.get("error")
        and (lens is None or r.get("prompt_len") == lens.get((r["condition"], r["qid"])))
    }


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


def check_corpus(questions: list[dict]) -> None:
    """Every gold table in every route, built from the very replies 23 saved (none changed since)
    by the renderer as it is now (no file left from before an ocr.py change)."""
    missing = [
        (route, g)
        for route in CONDITIONS.values()
        if route
        for q in questions
        for g in q["gold"]
        if not (HERE / "corpus" / route / f"{g}.txt").exists()
    ]
    built_from = json.loads((HERE / "corpus/sources.json").read_text("utf-8"))
    if missing or built_from != sources(RAW23):
        raise SystemExit(f"corpus incomplete or 23's replies changed, rebuild: {missing[:5]}")
    stale = [
        (route, tid)
        for route, tables in build(RAW23).items()
        for tid, text in tables.items()
        if (HERE / "corpus" / route / f"{tid}.txt").read_text("utf-8") != text
    ]
    if stale:
        raise SystemExit(f"corpus differs from a fresh build, rebuild: {stale[:5]}")


def load_corpus() -> dict[str, dict[str, str]]:
    """route -> 表號 -> text."""
    return {
        route: {p.stem: p.read_text("utf-8") for p in (HERE / "corpus" / route).glob("*.txt")}
        for route in (v for v in CONDITIONS.values() if v)
    }


def evidence_for(q: dict, fmt: str, corpus: dict[str, dict[str, str]]) -> tuple[str, str]:
    text = "\n\n".join(f"【{g}】\n{corpus[fmt][g]}" for g in q["gold"])
    return text, "、".join(q["gold"])


def prompt_lens(questions: list[dict], corpus: dict) -> dict[tuple[str, str], int]:
    """(condition, qid) -> length of the prompt the corpus builds now."""
    out = {}
    for c, fmt in CONDITIONS.items():
        for q in questions:
            text, label = evidence_for(q, fmt, corpus) if fmt else (None, "")
            out[(c, q["id"])] = len(user_prompt(q["question"], evidence=text, label=label))
    return out


def ask(model: str, condition: str, q: dict, corpus: dict) -> dict:
    fmt = CONDITIONS[condition]
    text, label = evidence_for(q, fmt, corpus) if fmt else (None, "")
    prompt = user_prompt(q["question"], evidence=text, label=label)
    row = {"model": model, "condition": condition, "qid": q["id"], "prompt_len": len(prompt)}
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
    conditions = list(CONDITIONS)
    check_corpus(questions)
    corpus = load_corpus()
    byid = {q["id"]: q for q in questions}
    done = done_keys(raw, prompt_lens(questions, corpus))
    todo = [k for k in grid(models, conditions, questions) if k not in done]
    logger.info(
        "grid",
        extra={"ctx": {"total": len(models) * len(conditions) * len(questions), "todo": len(todo)}},
    )
    with raw.open("a", encoding="utf-8") as f:
        for i, (m, c, qid) in enumerate(todo, 1):
            row = ask(m, c, byid[qid], corpus)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(todo):
                logger.info(
                    "progress",
                    extra={"ctx": {"done": i, "todo": len(todo), "model": m, "condition": c}},
                )

    errors = sum(1 for r in last_rows(raw) if r.get("error"))
    logger.info("failed calls left", extra={"ctx": {"n": errors}})
    first = {(r["model"], r["condition"], r["qid"]): r["content"] for r in last_rows(raw)}
    probes = []
    for m, c, qid in drift_sample(models, conditions, questions, per_cell=DRIFT_PER_CELL):
        again = ask(m, c, byid[qid], corpus)
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

    chars = {
        fmt: round(sum(len(t) for t in corpus[fmt].values()) / max(1, len(corpus[fmt])))
        for fmt in corpus
    }
    meta = {
        "date": date.today().isoformat(),
        "machine": platform.node(),
        "ollama": ollama.version(),
        "models": {m: ollama.model_digest(m) for m in models},
        "think": {m: THINK[m] for m in models},
        "options": OPTIONS,
        "pdfplumber": pkg_version("pdfplumber"),
        "conditions": CONDITIONS,
        "mean_chars_per_table": chars,
        "n_questions": {
            "total": len(questions),
            "tables": len({g for q in questions for g in q["gold"]}),
        },
        "drift": {"same": same, "n": len(probes), "rows": probes},
    }
    (RESULTS / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
