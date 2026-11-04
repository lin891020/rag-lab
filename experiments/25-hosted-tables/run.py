"""Claude answers 06's 82 table questions from four kinds of evidence -> results/raw.jsonl.

    uv run python experiments/25-hosted-tables/run.py [--models a,b] [--sample N] [--budget 25]

Conditions (each asked twice: the API takes no temperature, so the -2 run is the noise floor):
  clean     04's markdown of the PDF text layer, title and notes included
  glm2      glm-ocr's reply for 06's page rendered as 06 did (24's corpus), no title
  image     06's clean 300 dpi page(s) of the gold table, PNG as rendered, sent as is
  image-L2  22's worst scan of the same page(s) (150 dpi, 1.5°, blur, noise, JPEG q30)
The prompt is 06's; for the image conditions only the evidence block changes and the pages ride
before the text. Every call is billed into the running cost, failed ones too, and the run stops
once the cost passes --budget. A stop other than end_turn (refusal, max_tokens) is the model's
answer: kept as a final row, scored wrong. A failed call (network, server) is asked again on the
next run. Resumable: a final row counts as done only while its prompt's sha256 and its pages'
sha256s are the ones the inputs give now. Pages are checked against hashes git records (23's
reading of 06's pages, 22's scan manifest).
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
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
PNG06 = ROOT / "data/06-ocr/png"
L2 = ROOT / "data/22-scan/L2"
READ23 = HERE.parent / "23-ocr-size/corpus/glm2-s100-raw"  # <page>.json holds the PNG's sha256
SCAN22 = HERE.parent / "22-degraded-scan/results/scan_manifest.json"
sys.path.insert(0, str(HERE))
import claude  # noqa: E402
from prompts import IMAGES, SCHEMA, SYSTEM, TEXT, image_prompt, user_prompt  # noqa: E402

logger = logging.getLogger(__name__)

MODELS = ["claude-haiku-5-5", "claude-opus-5-5"]  # the cheap one first
BASE = [*TEXT, *IMAGES]
CONDITIONS = BASE + [f"{c}-2" for c in BASE]
BUDGET = 25.0  # US$; the plan says stop and ask Mike above it
CALL_ERRORS = (OSError, ValueError, anthropic.APIError)  # a KeyError is a bug: let it crash
MEDIA = {".png": "image/png", ".jpg": "image/jpeg"}


def grid(models: list[str], questions: list[dict]) -> Iterator[tuple[str, str, str]]:
    for m in models:
        for c in CONDITIONS:
            for q in questions:
                yield (m, c, q["id"])


def read_raw(raw: Path) -> list[dict]:
    if not raw.exists():
        return []
    return [json.loads(ln) for ln in raw.read_text("utf-8").splitlines() if ln.strip()]


def last_rows(raw: Path) -> list[dict]:
    """One row per (model, condition, qid): the last good one, else the last failed one."""
    best: dict[tuple[str, str, str], dict] = {}
    for r in read_raw(raw):
        k = (r["model"], r["condition"], r["qid"])
        if k not in best or not r.get("error") or best[k].get("error"):
            best[k] = r
    return list(best.values())


def done_keys(
    raw: Path, sig: dict[tuple[str, str], tuple[str, list[str]]]
) -> set[tuple[str, str, str]]:
    """Keys with a final row (an answer or a stop) asked with the prompt and pages the inputs give
    now."""
    return {
        (r["model"], r["condition"], r["qid"])
        for r in last_rows(raw)
        if not r.get("error")
        and (r.get("prompt_sha"), r.get("images", [])) == sig.get((r["condition"], r["qid"]))
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


def load_corpus() -> dict[str, dict[str, str]]:
    """route -> 表號 -> text."""
    return {
        route: {p.stem: p.read_text("utf-8") for p in (HERE / "corpus" / route).glob("*.txt")}
        for route in TEXT.values()
    }


def text_prompt(cond: str, q: dict, corpus: dict[str, dict[str, str]]) -> str:
    route = TEXT[cond.removesuffix("-2")]
    text = "\n\n".join(f"【{g}】\n{corpus[route][g]}" for g in q["gold"])  # 24's evidence_for
    return user_prompt(q["question"], evidence=text, label="、".join(q["gold"]))


def pages_for(cond: str, tid: str) -> list[Path]:
    base = cond.removesuffix("-2")
    if base == "image":
        return sorted(PNG06.glob(f"{tid}-p*.png"))
    if base == "image-L2":
        return sorted(L2.glob(f"{tid}-p*.jpg"))
    return []


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def recorded_hashes() -> dict[str, str]:
    """The pages' sha256 as git records them: 23 read 06's PNGs (s100 = the PNG as rendered) and
    22 wrote its scans' manifest."""
    out = {
        f"image/{p.stem}.png": json.loads(p.read_text("utf-8"))["sha256"]
        for p in READ23.glob("*.json")
    }
    for name, v in json.loads(SCAN22.read_text("utf-8"))["images"].items():
        if name.startswith("L2/"):
            out[f"image-L2/{name.removeprefix('L2/')}"] = v["sha256"]
    return out


def check_images(questions: list[dict]) -> dict[str, str]:
    """Every gold table has its pages in both sources, the same number, each the bytes git
    records. -> {"image/<file>" | "image-L2/<file>": sha256}."""
    recorded = recorded_hashes()
    shas, bad = {}, []
    for tid in sorted({g for q in questions for g in q["gold"]}):
        clean, scan = pages_for("image", tid), pages_for("image-L2", tid)
        if not clean or len(clean) != len(scan):
            bad.append((tid, len(clean), len(scan)))
        for key, p in [(f"image/{p.name}", p) for p in clean] + [
            (f"image-L2/{p.name}", p) for p in scan
        ]:
            shas[key] = _sha(p)
            if recorded.get(key) != shas[key]:
                bad.append((key, "sha256 differs from the one git records"))
    if bad:
        raise SystemExit(f"page images missing or changed: {bad[:5]}")
    return shas


def prompt_and_pages(cond: str, q: dict, corpus: dict) -> tuple[str, list[Path]]:
    pages = [p for g in q["gold"] for p in pages_for(cond, g)]
    if not pages:
        return text_prompt(cond, q, corpus), []
    return image_prompt(q["question"], label="、".join(q["gold"]), pages=len(pages)), pages


def _text_sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def signatures(questions: list[dict]) -> dict[tuple[str, str], tuple[str, list[str]]]:
    """(condition, qid) -> (sha256 of the prompt, sha256 of each page sent) for the inputs as
    they are."""
    corpus, out = load_corpus(), {}
    for c in CONDITIONS:
        for q in questions:
            prompt, pages = prompt_and_pages(c, q, corpus)
            out[(c, q["id"])] = (_text_sha(prompt), [_sha(p) for p in pages])
    return out


def ask(model: str, cond: str, q: dict, corpus: dict) -> dict:
    prompt, pages = prompt_and_pages(cond, q, corpus)
    images = [(p.read_bytes(), MEDIA[p.suffix]) for p in pages]
    row: dict = {
        "model": model,
        "condition": cond,
        "qid": q["id"],
        "run": 2 if cond.endswith("-2") else 1,
        "date": date.today().isoformat(),
        "prompt_len": len(prompt),
        "prompt_sha": _text_sha(prompt),
        "images": [hashlib.sha256(b).hexdigest() for b, _ in images],
    }
    try:
        r = claude.chat(prompt, model=model, system=SYSTEM, schema=SCHEMA, images=images)
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
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--sample", type=int, default=0, help="n questions spread over the list")
    ap.add_argument("--budget", type=float, default=BUDGET, help="stop once passed (US$)")
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args(argv)
    models = args.models.split(",")
    setup_logging("rag_lab")
    claude.load_env()
    questions = load_questions()
    if args.sample:
        questions = sample(questions, args.sample)
    shas = check_images(questions)
    corpus = load_corpus()
    sig = signatures(questions)
    byid = {q["id"]: q for q in questions}
    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.out / "raw.jsonl"
    spent = spent_usd(read_raw(raw))
    done = done_keys(raw, sig)
    todo = [k for k in grid(models, questions) if k not in done]
    logger.info("grid", extra={"ctx": {"todo": len(todo), "spent_usd": round(spent, 4)}})
    if spent > args.budget:
        logger.error("budget already passed", extra={"ctx": {"spent_usd": spent}})
        return 2
    with raw.open("a", encoding="utf-8") as f:
        for i, (m, c, qid) in enumerate(todo, 1):
            row = ask(m, c, byid[qid], corpus)
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
        "models": models,
        "served_models": sorted({r.get("served_model", "") for r in good}),
        "effort": claude.EFFORT,
        "max_tokens": claude.MAX_TOKENS,
        "conditions": CONDITIONS,
        "n_questions": {"total": len(questions)},
        "images_sha256": shas,
        "tokens": {
            m: [
                sum(r["input_tokens"] for r in every if r["model"] == m),
                sum(r["output_tokens"] for r in every if r["model"] == m),
            ]
            for m in models
        },
        "calls": len(every),
        "failed_calls": sum(bool(r.get("error")) for r in every),
        "failed_left": sum(bool(r.get("error")) for r in rows),
        "stopped": sum(bool(r.get("stopped")) for r in rows),
        "spent_usd": round(spent_usd(every), 4),
        "prices_usd_per_mtok": {m: claude.PRICES[m] for m in models},
    }
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
