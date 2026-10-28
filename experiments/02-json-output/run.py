"""Run the grid (models x conditions x questions) against the host ollama -> results/raw.jsonl.

    uv run python experiments/02-json-output/run.py [--models qwen3:8b,gpt-oss:20b]

Resumable: rows already in raw.jsonl are skipped, so a crash or Ctrl-C costs nothing. After the
grid, a small drift sample is re-asked and compared (meta.json: drift). Gold text comes from the
01 corpus (../01-chunking/corpus); it is not copied here.
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
from rag_lab.pdf import extract_text

HERE = Path(__file__).parent
ONE = HERE.parent / "01-chunking"
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ONE))
from chunkers import parse_articles  # noqa: E402
from prompts import CONDITIONS, SYSTEM, user_prompt  # noqa: E402

logger = logging.getLogger(__name__)

MODELS = ["qwen3:8b", "gpt-oss:20b"]
# gpt-oss ignores think=False (keeps thinking; under a schema that empties the content), so it
# runs at the lowest level it accepts. Same setting for every condition of that model.
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
    """First `per_cell` questions of every cell: deterministic, no RNG needed."""
    ids = [q["id"] for q in questions[:per_cell]]
    return [(m, c, q) for m in models for c in conditions for q in ids]


def load_questions() -> list[dict]:
    return [
        json.loads(ln)
        for ln in (HERE / "questions.jsonl").read_text("utf-8").splitlines()
        if ln.strip()
    ]


def load_gold() -> tuple[dict[str, str], dict[str, str]]:
    arts = {str(a.num): a.body for a in parse_articles(extract_text(ONE / "corpus/law.pdf"))}
    tabs = {p.stem: p.read_text("utf-8") for p in (ONE / "corpus/tables").glob("*.txt")}
    return arts, tabs


def evidence_for(q: dict, arts: dict[str, str], tabs: dict[str, str]) -> tuple[str, str]:
    src = arts if q["type"] == "article" else tabs
    text = "\n\n".join(f"【{g}】\n{src[g]}" for g in q["gold"])
    label = "、".join(("第 " + g + " 條") if q["type"] == "article" else g for g in q["gold"])
    return text, label


def ask(model: str, condition: str, q: dict, arts: dict, tabs: dict) -> dict:
    spec = CONDITIONS[condition]
    text, label = evidence_for(q, arts, tabs) if spec["evidence"] else (None, "")
    prompt = user_prompt(q["question"], evidence=text, label=label)
    row = {"model": model, "condition": condition, "qid": q["id"]}
    try:
        r = ollama.chat(
            prompt,
            model=model,
            system=SYSTEM,
            format=spec["format"],
            think=THINK[model],
            options=OPTIONS,
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
    arts, tabs = load_gold()
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
            row = ask(m, c, byid[qid], arts, tabs)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(todo):
                logger.info(
                    "progress",
                    extra={"ctx": {"done": i, "todo": len(todo), "model": m, "condition": c}},
                )

    # drift probe: ask a few cells again, compare content
    first = {
        (r["model"], r["condition"], r["qid"]): r["content"]
        for r in map(json.loads, raw.read_text("utf-8").splitlines())
        if r
    }
    probes = []
    for m, c, qid in drift_sample(models, conditions, questions, per_cell=DRIFT_PER_CELL):
        again = ask(m, c, byid[qid], arts, tabs)
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

    meta = {
        "date": date.today().isoformat(),
        "machine": platform.node(),
        "ollama": ollama.version(),
        "models": {m: ollama.model_digest(m) for m in models},
        "think": {m: THINK[m] for m in models},
        "options": OPTIONS,
        "conditions": {
            c: ("schema" if isinstance(s["format"], dict) else s["format"])
            for c, s in CONDITIONS.items()
        },
        "n_questions": {
            "total": len(questions),
            "article": sum(q["type"] == "article" for q in questions),
            "table": sum(q["type"] == "table" for q in questions),
        },
        "drift": {"same": same, "n": len(probes), "rows": probes},
    }
    (RESULTS / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
