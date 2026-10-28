"""17 tables (03's set) -> page images are in data/06-ocr/png (rendered at 300 dpi by
render.py); this script runs the OCR routes and writes corpus/<route>/<表號>.txt, plus
corpus/clean/<表號>.txt copied from 04's markdown (the pdfplumber reading = ground truth).

    uv run python experiments/06-ocr-source/build_corpus.py [--routes vision,tesseract,glm,glm2] [--from-raw]

The glm route talks to the host ollama (glm-ocr); the other two are local CPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from rag_lab.log import setup_logging

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
PNG = ROOT / "data/06-ocr/png"
FOUR = HERE.parent / "04-table-lookup"
CORPUS = HERE / "corpus"
sys.path.insert(0, str(HERE))
from ocr import (  # noqa: E402
    GLM_PROMPT,
    GLM_PROMPT_NL,
    glm_ocr,
    glm_to_markdown,
    segments_to_markdown,
    tesseract_segments,
    vision_segments,
)

logger = logging.getLogger(__name__)
ROUTES = ("vision", "tesseract", "glm", "glm2")


def table_ids() -> list[str]:
    return sorted(p.stem for p in (FOUR / "corpus/md").glob("*.txt"))


def pages_of(tid: str) -> list[Path]:
    return sorted(PNG.glob(f"{tid}-p*.png"))


def run_route(route: str, tid: str, *, from_raw: bool = False) -> tuple[str, dict]:
    """Markdown for one table (pages concatenated) and bookkeeping."""
    parts, info = [], {"pages": 0, "seconds": 0.0}
    for png in pages_of(tid):
        t0 = time.time()
        if route == "vision":
            parts.append(segments_to_markdown(vision_segments(png)))
        elif route == "tesseract":
            parts.append(segments_to_markdown(tesseract_segments(png)))
        else:
            cached = CORPUS / f"{route}-raw" / f"{png.stem}.html"
            if from_raw and cached.exists():
                raw = cached.read_text("utf-8")
            else:
                raw = glm_ocr(png, prompt=GLM_PROMPT_NL if route == "glm2" else GLM_PROMPT)
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_text(raw, encoding="utf-8")
            parts.append(glm_to_markdown(raw))
        info["pages"] += 1
        info["seconds"] += time.time() - t0
    return "\n".join(parts), info


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", default=",".join(ROUTES))
    ap.add_argument("--from-raw", action="store_true", help="glm: reuse corpus/glm-raw replies")
    args = ap.parse_args(argv)
    setup_logging("rag_lab")
    stats_path = CORPUS / "stats.json"
    stats = json.loads(stats_path.read_text("utf-8")) if stats_path.exists() else {}
    (CORPUS / "clean").mkdir(parents=True, exist_ok=True)
    for tid in table_ids():
        (CORPUS / "clean" / f"{tid}.txt").write_text(
            (FOUR / "corpus/md" / f"{tid}.txt").read_text("utf-8"), encoding="utf-8"
        )
    for route in args.routes.split(","):
        (CORPUS / route).mkdir(parents=True, exist_ok=True)
        for tid in table_ids():
            text, info = run_route(route, tid, from_raw=args.from_raw)
            (CORPUS / route / f"{tid}.txt").write_text(text, encoding="utf-8")
            stats.setdefault(route, {})[tid] = {**info, "chars": len(text)}
        logger.info(
            "route done",
            extra={
                "ctx": {
                    "route": route,
                    "tables": len(stats[route]),
                    "seconds": round(sum(v["seconds"] for v in stats[route].values()), 1),
                }
            },
        )
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=1), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
