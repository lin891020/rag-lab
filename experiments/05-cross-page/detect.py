"""How often does the continuation rule fire when it should, and when it should not?

    uv run python experiments/05-cross-page/detect.py   -> results/detect.json

Positives: every (page N, page N+1) pair inside one attachment PDF that has a table on both
pages (the 4 real continuations). Negatives: the last page of attachment i followed by the
first page of attachment i+1, renumbered as if the two were one document (what a parser sees in
a concatenated PDF). Also records, per rejected pair, which check rejected it first.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
from pathlib import Path

from rag_lab.log import setup_logging

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
ATTACH = ROOT / "data/01-chunking/attachments"
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE))
from tables import (  # noqa: E402
    BOTTOM_TH,
    EDGE_MATCH,
    TOP_TH,
    WIDTH_TOL,
    PageGrid,
    _edges_match,
    is_continuation,
    load_pages,
)

logger = logging.getLogger(__name__)

POSITIVES = {("0000395929", 1), ("0000395996", 1), ("0000395998", 1), ("0000395998", 2)}


def first_failing_check(prev: PageGrid, cur: PageGrid) -> str:
    """Name of the first check in is_continuation's order that rejects the pair, or ''."""
    if prev.bbox[3] < BOTTOM_TH * prev.page_height or prev.words_below:
        return "prev_not_at_bottom"
    if cur.bbox[1] > TOP_TH * cur.page_height or cur.words_above or cur.has_title:
        return "cur_has_title_or_not_at_top"
    w_prev, w_cur = prev.bbox[2] - prev.bbox[0], cur.bbox[2] - cur.bbox[0]
    if abs(w_prev - w_cur) > WIDTH_TOL * max(w_prev, w_cur):
        return "width"
    if _edges_match(prev.x_edges, cur.x_edges) < EDGE_MATCH:
        return "column_edges"
    return ""


def pairs(grids: dict[str, list[PageGrid]]) -> list[tuple[str, PageGrid, PageGrid, bool]]:
    out = []
    fids = sorted(grids)
    for fid in fids:
        gs = grids[fid]
        for a, b in zip(gs, gs[1:], strict=False):
            out.append((f"{fid}:p{a.page_no}->p{b.page_no}", a, b, (fid, b.page_no) in POSITIVES))
    for fa, fb in zip(fids, fids[1:], strict=False):
        a, b = grids[fa][-1], grids[fb][0]
        b2 = dataclasses.replace(b, page_no=a.page_no + 1)
        out.append((f"{fa}:p{a.page_no}->{fb}:p{b.page_no}", a, b2, False))
    return out


def evaluate(grids: dict[str, list[PageGrid]]) -> dict:
    rows = []
    for name, a, b, truth in pairs(grids):
        pred = is_continuation(a, b)
        rows.append(
            {
                "pair": name,
                "truth": truth,
                "pred": pred,
                "rejected_by": first_failing_check(a, b),
                "edge_match": round(_edges_match(a.x_edges, b.x_edges), 3),
            }
        )
    tp = sum(r["truth"] and r["pred"] for r in rows)
    fp = sum((not r["truth"]) and r["pred"] for r in rows)
    fn = sum(r["truth"] and not r["pred"] for r in rows)
    tn = sum((not r["truth"]) and not r["pred"] for r in rows)
    by_check: dict[str, int] = {}
    for r in rows:
        if not r["truth"]:
            by_check[r["rejected_by"] or "accepted"] = (
                by_check.get(r["rejected_by"] or "accepted", 0) + 1
            )
    # how many negatives the geometry alone (no title check) would still reject
    geo_only = sum(
        1
        for name, a, b, truth in pairs(grids)
        if not truth
        and (
            a.bbox[3] < BOTTOM_TH * a.page_height
            or b.bbox[1] > TOP_TH * b.page_height
            or abs((a.bbox[2] - a.bbox[0]) - (b.bbox[2] - b.bbox[0]))
            > WIDTH_TOL * max(a.bbox[2] - a.bbox[0], b.bbox[2] - b.bbox[0])
            or _edges_match(a.x_edges, b.x_edges) < EDGE_MATCH
        )
    )
    return {
        "n_pairs": len(rows),
        "positives": tp + fn,
        "negatives": fp + tn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "negatives_rejected_by_first_check": by_check,
        "negatives_rejected_by_geometry_alone": geo_only,
        "rows": rows,
    }


def main() -> int:
    setup_logging("rag_lab")
    grids = {}
    for p in sorted(ATTACH.glob("*.pdf")):
        gs = load_pages(p)
        if gs:
            grids[p.stem] = gs
    out = evaluate(grids)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "detect.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    logger.info(
        "detect",
        extra={
            "ctx": {
                k: out[k]
                for k in ("n_pairs", "tp", "fp", "fn", "tn", "negatives_rejected_by_geometry_alone")
            }
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
