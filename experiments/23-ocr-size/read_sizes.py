"""Every page size -> corpus/<route>/<表號>.txt (pages concatenated) read by glm-ocr, raw replies
in corpus/<route>-raw/<page>.html (reused on resume, only for the same image), bookkeeping in
corpus/stats.json (seconds; per page: reply format, image sha256 and size, done_reason,
eval_count and prompt_eval_count = the prompt's tokens, nearly all of them the image's), plus
the glm-ocr digest and ollama version under "_versions".

    uv run python experiments/23-ocr-size/read_sizes.py [--routes glm2-s50,...] [--tables 表九二 --limit-pages 1 --corpus <scratch dir>]

glm-ocr talks to the host ollama with 06's prompt (trailing newline) and settings, as in 22.
corpus/clean (04's markdown) was copied from 22.
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

from rag_lab import ollama
from rag_lab.log import setup_logging

HERE = Path(__file__).parent
FOUR = HERE.parent / "04-table-lookup"
CORPUS = HERE / "corpus"
sys.path.insert(0, str(HERE))
from ocr import GLM_PROMPT_NL, glm_ocr, glm_to_markdown  # noqa: E402
from shrink import PNG06, SCALES, SHRUNK  # noqa: E402

logger = logging.getLogger(__name__)
ROUTES = {"glm2-s100": 1.0} | {f"glm2-{k}": v for k, v in SCALES.items()}


def page_dir(route: str) -> Path:
    scale = route.removeprefix("glm2-")
    return PNG06 if scale == "s100" else SHRUNK / scale


def table_ids() -> list[str]:
    return sorted(p.stem for p in (FOUR / "corpus/md").glob("*.txt"))


def pages_of(route: str, tid: str) -> list[Path]:
    return sorted(page_dir(route).glob(f"{tid}-p*.png"))


def reply_format(raw: str) -> str:
    if "<table" in raw:
        return "html"
    lines = raw.splitlines()
    if any(ln.lstrip().startswith("|") for ln in lines):
        return "markdown"
    if any("\t" in ln for ln in lines):
        return "tsv"
    return "plain"


def read_page(img: Path, raw_dir: Path) -> tuple[str, dict]:
    """One page -> (markdown, info). The reply is cached with the image's sha256 and reused only
    for that same image."""
    sha = hashlib.sha256(img.read_bytes()).hexdigest()
    cached, side = raw_dir / f"{img.stem}.html", raw_dir / f"{img.stem}.json"
    if cached.exists():
        info = json.loads(side.read_text("utf-8")) if side.exists() else {}
        if info.get("sha256") != sha:
            raise SystemExit(f"{cached} was read from another image than {img}: delete it")
        return glm_to_markdown(cached.read_text("utf-8")), {**info, "cached": True}
    r = glm_ocr(img, prompt=GLM_PROMPT_NL)
    raw = r["content"]
    info = {k: r[k] for k in ("done_reason", "eval_count", "prompt_eval_count")}
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
        if stats.get(route, {}).get(tid, {}).get("pages") != n_pages(route, tid)
    ]
    if bad:
        raise SystemExit(f"corpus incomplete or partial, rebuild first: {bad[:5]} ({len(bad)})")


def run_route(route: str, tid: str, corpus: Path, *, limit_pages: int = 0) -> tuple[str, dict]:
    pages = pages_of(route, tid)
    if not pages:
        raise FileNotFoundError(f"no {route} images for {tid}: run shrink.py first")
    if limit_pages:
        pages = pages[:limit_pages]
    parts, info = [], {"pages": 0, "seconds": 0.0, "per_page": {}}
    for img in pages:
        t0 = time.time()
        md, page_info = read_page(img, corpus / f"{route}-raw")
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
    corpus = args.corpus
    stats_path = corpus / "stats.json"
    stats = json.loads(stats_path.read_text("utf-8")) if stats_path.exists() else {}
    stats["_versions"] = {"glm-ocr": ollama.model_digest("glm-ocr"), "ollama": ollama.version()}
    tids = args.tables.split(",") if args.tables else table_ids()
    try:
        for route in args.routes.split(","):
            (corpus / route).mkdir(parents=True, exist_ok=True)
            for tid in tids:
                text, info = run_route(route, tid, corpus, limit_pages=args.limit_pages)
                (corpus / route / f"{tid}.txt").write_text(text, encoding="utf-8")
                stats.setdefault(route, {})[tid] = {**info, "chars": len(text)}
            secs = sum(v["seconds"] for v in stats[route].values())
            logger.info(
                "route done",
                extra={
                    "ctx": {"route": route, "tables": len(stats[route]), "seconds": round(secs)}
                },
            )
    finally:
        corpus.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=1), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
