"""Every route x level -> corpus/<route>/<表號>.txt (pages concatenated), raw replies in
corpus/<route>-raw/<page>.html (glm2, opus: reused on resume, so nothing is asked or paid twice),
bookkeeping in corpus/stats.json (seconds; per page: reply format, image sha256 and size,
glm2's done_reason, opus' tokens) and every paid opus call in corpus/opus-ledger.jsonl.

    uv run python experiments/22-degraded-scan/build_corpus.py [--routes glm2-L0,...] [--tables 表五二 --limit-pages 1 --corpus <scratch dir>]

L0 = 06's clean 300 dpi PNGs (data/06-ocr/png); L1/L2 = degrade.py's JPEGs (data/22-scan).
glm2 talks to the host ollama (glm-ocr, 06's prompt with the trailing newline); vision is local;
opus is the Claude API (key from .env), stopped before a call once BUDGET is spent.
corpus/clean (04's markdown) and corpus/glm2-06 (06's glm2 reading) were copied from 06.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

from PIL import Image

from rag_lab.log import setup_logging

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
PNG06 = ROOT / "data/06-ocr/png"
SCAN = ROOT / "data/22-scan"
FOUR = HERE.parent / "04-table-lookup"
CORPUS = HERE / "corpus"
sys.path.insert(0, str(HERE))
import claude_ocr  # noqa: E402
from ocr import (  # noqa: E402
    GLM_PROMPT_NL,
    glm_ocr,
    glm_to_markdown,
    segments_to_markdown,
    vision_segments,
)

logger = logging.getLogger(__name__)
ROUTES = {
    "glm2-L0": ("glm2", "L0"),  # 06's glm2 asked again: the OCR noise floor
    "glm2-L1": ("glm2", "L1"),
    "glm2-L2": ("glm2", "L2"),
    "vision-L1": ("vision", "L1"),
    "vision-L2": ("vision", "L2"),
    "opus-L1": ("opus", "L1"),
    "opus-L2": ("opus", "L2"),
}
BUDGET = 5.0  # US$ for the opus route; the plan says stop and ask above it
LEDGER = "opus-ledger.jsonl"  # one line per paid call, written as it returns


def page_dir(level: str) -> tuple[Path, str]:
    return (PNG06, "png") if level == "L0" else (SCAN / level, "jpg")


def table_ids() -> list[str]:
    return sorted(p.stem for p in (FOUR / "corpus/md").glob("*.txt"))


def pages_of(level: str, tid: str) -> list[Path]:
    d, ext = page_dir(level)
    return sorted(d.glob(f"{tid}-p*.{ext}"))


def reply_format(raw: str) -> str:
    if "<table" in raw:
        return "html"
    lines = raw.splitlines()
    if any(ln.lstrip().startswith("|") for ln in lines):
        return "markdown"
    if any("\t" in ln for ln in lines):
        return "tsv"
    return "plain"


def ledger_usd(ledger: Path) -> float:
    """US$ of every opus call ever made into this corpus (one line per call, stopped ones too)."""
    if not ledger.exists():
        return 0.0
    rows = [json.loads(ln) for ln in ledger.read_text("utf-8").splitlines() if ln.strip()]
    return sum(
        claude_ocr.price(claude_ocr.OCR_MODEL, r["tokens_in"], r["tokens_out"]) for r in rows
    )


def _log_call(ledger: Path, img: Path, tokens_in: int, tokens_out: int, stop: str) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as f:
        row = {"page": img.stem, "level": img.parent.name, "tokens_in": tokens_in}
        f.write(json.dumps({**row, "tokens_out": tokens_out, "stop": stop}) + "\n")


def _ask(kind: str, img: Path, ledger: Path) -> tuple[str, dict]:
    if kind == "glm2":
        r = glm_ocr(img, prompt=GLM_PROMPT_NL)
        return r["content"], {"done_reason": r["done_reason"], "eval_count": r["eval_count"]}
    spent = ledger_usd(ledger)
    if spent >= BUDGET:
        raise SystemExit(f"opus spent US${spent:.2f} >= budget {BUDGET}: ask Mike")
    media = "image/png" if img.suffix == ".png" else "image/jpeg"
    try:
        r = claude_ocr.ocr(img.read_bytes(), media_type=media)
    except claude_ocr.StopError as exc:
        _log_call(ledger, img, exc.tokens_in, exc.tokens_out, str(exc))
        raise
    _log_call(ledger, img, r["tokens_in"], r["tokens_out"], "end_turn")
    info = {k: r[k] for k in ("tokens_in", "tokens_out", "served_model", "request_id")}
    return r["content"], info


def read_page(kind: str, img: Path, raw_dir: Path, ledger: Path) -> tuple[str, dict]:
    """One page -> (markdown, info). glm2/opus replies are cached with the image's sha256 and
    reused only for that same image; opus calls are written to the ledger as they return."""
    if kind == "vision":
        return segments_to_markdown(vision_segments(img)), {"format": "rows"}
    sha = hashlib.sha256(img.read_bytes()).hexdigest()
    cached, side = raw_dir / f"{img.stem}.html", raw_dir / f"{img.stem}.json"
    if cached.exists():
        info = json.loads(side.read_text("utf-8")) if side.exists() else {}
        if info.get("sha256") != sha:
            raise SystemExit(f"{cached} was read from another image than {img}: delete it")
        return glm_to_markdown(cached.read_text("utf-8")), {**info, "cached": True}
    raw, info = _ask(kind, img, ledger)
    info.update(sha256=sha, size=list(Image.open(img).size), format=reply_format(raw))
    raw_dir.mkdir(parents=True, exist_ok=True)
    cached.write_text(raw, encoding="utf-8")
    side.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    return glm_to_markdown(raw), info


def check_corpus(routes, stats: dict, tids: list[str], n_pages) -> None:
    """Every route built for every table from all of its pages (a trial run reads fewer)."""
    bad = [
        (route, tid)
        for route in routes
        for tid in tids
        if stats.get(route, {}).get(tid, {}).get("pages") != n_pages(ROUTES[route][1], tid)
    ]
    if bad:
        raise SystemExit(f"corpus incomplete or partial, rebuild first: {bad[:5]} ({len(bad)})")


def run_route(route: str, tid: str, corpus: Path, *, limit_pages: int = 0) -> tuple[str, dict]:
    kind, level = ROUTES[route]
    pages = pages_of(level, tid)
    if not pages:
        raise FileNotFoundError(f"no {level} images for {tid}: run degrade.py first")
    if limit_pages:
        pages = pages[:limit_pages]
    parts, info = [], {"pages": 0, "seconds": 0.0, "per_page": {}}
    for img in pages:
        t0 = time.time()
        md, page_info = read_page(kind, img, corpus / f"{route}-raw", corpus / LEDGER)
        parts.append(md)
        info["pages"] += 1
        info["seconds"] += time.time() - t0
        info["per_page"][img.stem] = page_info
    return "\n".join(parts), info


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", default=",".join(ROUTES))
    ap.add_argument("--tables", default="", help="comma-separated 表號 (trial run)")
    ap.add_argument("--limit-pages", type=int, default=0)
    ap.add_argument("--corpus", type=Path, default=CORPUS, help="a trial run writes elsewhere")
    args = ap.parse_args(argv)
    setup_logging("rag_lab")
    claude_ocr.load_env()
    corpus = args.corpus
    stats_path = corpus / "stats.json"
    stats = json.loads(stats_path.read_text("utf-8")) if stats_path.exists() else {}
    tids = args.tables.split(",") if args.tables else table_ids()
    try:
        for route in args.routes.split(","):
            (corpus / route).mkdir(parents=True, exist_ok=True)
            for tid in tids:
                text, info = run_route(route, tid, corpus, limit_pages=args.limit_pages)
                (corpus / route / f"{tid}.txt").write_text(text, encoding="utf-8")
                stats.setdefault(route, {})[tid] = {**info, "chars": len(text)}
            logger.info(
                "route done",
                extra={
                    "ctx": {
                        "route": route,
                        "tables": len(stats[route]),
                        "seconds": round(sum(v["seconds"] for v in stats[route].values()), 1),
                        "opus_usd": round(ledger_usd(corpus / LEDGER), 4),
                    }
                },
            )
    finally:
        corpus.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=1), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
