"""How much of the table survived OCR, before any model sees it.

    uv run python experiments/06-ocr-source/structure.py   -> results/structure.csv

Ground truth = 04's corpus/tables.json (pdfplumber grid, checked against the PDF images in 03).
Per table and route:
  cell_recall     share of non-empty ground-truth data cells that appear as a whole cell
                  (|-delimited) somewhere in the OCR markdown, after norm()
  num_recall      the same, restricted to cells that are numbers (the ones questions ask for)
  row_recall      share of ground-truth rows whose cells all appear, in order, inside one OCR row
  text_recall     share of ground-truth cells whose text appears anywhere in the OCR text
                  (substring) -- what a reader could still find without the grid
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
FOUR = HERE.parent / "04-table-lookup"
CORPUS = HERE / "corpus"
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE))
from ocr import norm  # noqa: E402

logger = logging.getLogger(__name__)
_NUM = re.compile(r"^[\d.,\-–~～以上下超過至\s]+$")


def ocr_rows(md: str) -> list[list[str]]:
    rows = []
    for ln in md.splitlines():
        if ln.startswith("|") and not ln.startswith("|---"):
            rows.append([norm(c) for c in ln.strip("|").split("|")])
    return rows


def _in_order(needles: list[str], hay: list[str]) -> bool:
    i = 0
    for cell in hay:
        if i < len(needles) and cell == needles[i]:
            i += 1
    return i == len(needles)


def compare(gt: dict, md: str) -> dict[str, float]:
    rows = ocr_rows(md)
    cells = {c for r in rows for c in r if c}
    text = norm(md)
    gt_rows = [[norm(c) for c in r] for r in gt["rows"]]
    gt_cells = [c for r in gt_rows for c in r if c]
    nums = [c for c in gt_cells if _NUM.match(c) and any(ch.isdigit() for ch in c)]
    uniq_rows = [sorted(set(c for c in r if c), key=r.index) for r in gt_rows]
    return {
        "cell_recall": sum(c in cells for c in gt_cells) / max(1, len(gt_cells)),
        "num_recall": sum(c in cells for c in nums) / max(1, len(nums)),
        "row_recall": sum(any(_in_order(r, row) for row in rows) for r in uniq_rows)
        / max(1, len(uniq_rows)),
        "text_recall": sum(c in text for c in gt_cells) / max(1, len(gt_cells)),
        "n_cells": len(gt_cells),
        "n_nums": len(nums),
    }


def main() -> int:
    from rag_lab.log import setup_logging

    setup_logging("rag_lab")
    gt = json.loads((FOUR / "corpus/tables.json").read_text("utf-8"))
    out = []
    for route in sorted(p.name for p in CORPUS.iterdir() if p.is_dir() and p.name != "glm-raw"):
        for tid in sorted(gt):
            p = CORPUS / route / f"{tid}.txt"
            if not p.exists():
                continue
            out.append({"route": route, "table": tid, **compare(gt[tid], p.read_text("utf-8"))})
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "structure.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    by_route: dict[str, list[dict]] = {}
    for r in out:
        by_route.setdefault(r["route"], []).append(r)
    for route, rows in by_route.items():
        logger.info(
            "structure",
            extra={
                "ctx": {
                    "route": route,
                    **{
                        k: round(sum(r[k] for r in rows) / len(rows), 3)
                        for k in ("cell_recall", "num_recall", "row_recall", "text_recall")
                    },
                }
            },
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
