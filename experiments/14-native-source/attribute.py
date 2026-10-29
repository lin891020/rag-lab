"""Why did a question miss? One cause per (cell, question), mapped to 08's lookup table.

    uv run python experiments/14-native-source/attribute.py   -> results/attribution.csv,
                                                             results/attribution_rows.jsonl

Causes, checked in this order (the first that applies):
  no_chunks         the document produced no chunk in this cell
  garbage           the quote is not in the route's text and the document has (cid:N) pages
  no_text           … and the document has pages without a text layer: 08's 「掃描頁」
  rotated           … and the document has rotated pages (pdfplumber reads them backwards)
  extract_other     … none of the above explains the missing quote
  preamble_dropped  the quote is in the text but before the first article header, and the
                    chunking keeps articles only
  chunk_cut         the quote is in the text but no single chunk holds it whole: 08's 「塊界」
  over_budget       a gold chunk is in the top five but does not fit the 1,000-character budget
                    (a 1,000-character chunk plus its header or 《law》 prefix): 08's 「塊界」
                    in a new costume
  wrong_doc         a gold chunk exists; none of the top five chunks is from the document:
                    08's 「撈錯份」
  ranked_low        a gold chunk exists and the document is in the top five, but not it
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
CORPUS = HERE / "corpus"
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE))
from chunkers import parse_points  # noqa: E402
from run_retrieval import contains, norm  # noqa: E402

logger = logging.getLogger(__name__)

PREDICTED_BY_08 = {"no_chunks", "no_text", "chunk_cut", "over_budget", "wrong_doc"}


def classify(
    *,
    hit: bool,
    doc_chunks: int,
    gold_chunks: int,
    quote_in_text: bool,
    quote_before_first_article: bool,
    chunking: str,
    health: dict,
    top5_docs: list[str],
    doc: str,
    hit5: bool = False,
) -> str | None:
    if hit:
        return None
    if doc_chunks == 0:
        return "no_chunks"
    if gold_chunks == 0:
        if not quote_in_text:
            if health.get("garbage_pages", 0) > 0:
                return "garbage"
            if health.get("text_pages", 0) < health.get("pages", 0):
                return "no_text"
            if health.get("rotated_pages", 0) > 0:
                return "rotated"
            return "extract_other"
        if chunking in ("points", "rows") and quote_before_first_article:
            return "preamble_dropped"
        return "chunk_cut"
    if hit5:
        return "over_budget"
    if doc not in top5_docs:
        return "wrong_doc"
    return "ranked_low"


def quote_position(text: str, quote: str, chunking: str) -> tuple[bool, bool]:
    """(quote in text, quote before the first article header of this chunking's parser)."""
    if not contains(text, quote):
        return False, False
    arts = parse_points(text)
    if not arts:
        return True, False  # no 點 at all: points falls back to fixed-300, nothing is dropped
    head = norm(text[: arts[0].start])
    return True, norm(quote) in head


def main() -> int:
    from rag_lab.log import setup_logging

    setup_logging("rag_lab")
    docs = json.loads((CORPUS / "manifest.json").read_text("utf-8"))["docs"]
    health = {d["sysid"]: d["health"] for d in docs}
    questions = {
        json.loads(ln)["id"]: json.loads(ln)
        for ln in (HERE / "questions.jsonl").read_text("utf-8").splitlines()
        if ln.strip()
    }
    gold_counts = json.loads((RESULTS / "retrieval/gold_counts.json").read_text("utf-8"))
    raw = [
        json.loads(ln)
        for ln in (RESULTS / "retrieval/raw.jsonl").read_text("utf-8").splitlines()
        if ln.strip()
    ]
    texts: dict[tuple[str, str], str] = {}
    pos_cache: dict[tuple, tuple[bool, bool]] = {}
    rows = []
    counts: Counter[tuple[str, str]] = Counter()
    for r in raw:
        if r["retriever"] == "random":
            continue
        q = questions[r["qid"]]
        key = f"{r['route']}/{r['chunking']}/{'prefix' if r['prefix'] else 'noprefix'}"
        gc = gold_counts[key][r["qid"]]
        tk = (r["route"], q["doc"])
        if tk not in texts:
            texts[tk] = (CORPUS / "text" / f"{q['doc']}.{r['route']}.txt").read_text("utf-8")
        pk = (r["route"], r["chunking"], r["qid"])
        if pk not in pos_cache:
            pos_cache[pk] = quote_position(texts[tk], q["quote"], r["chunking"])
        in_text, before = pos_cache[pk]
        cause = classify(
            hit=r["hit@1000"] >= 1,
            doc_chunks=gc["doc_chunks"],
            gold_chunks=gc["gold_chunks"],
            quote_in_text=in_text,
            quote_before_first_article=before,
            chunking=r["chunking"],
            health=health[q["doc"]],
            top5_docs=r["top5_docs"],
            doc=q["doc"],
            hit5=r["hit@5"] >= 1,
        )
        if cause is None:
            continue
        cell = f"{key}/{r['retriever']}"
        counts[(cell, cause)] += 1
        rows.append(
            {"cell": cell, "qid": r["qid"], "doc": q["doc"], "type": q["type"], "cause": cause}
        )
    out = [
        {"cell": cell, "cause": cause, "n": n, "predicted_by_08": cause in PREDICTED_BY_08}
        for (cell, cause), n in sorted(counts.items())
    ]
    with (RESULTS / "attribution.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    (RESULTS / "attribution_rows.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    logger.info(
        "attributed", extra={"ctx": {"misses": len(rows), "cells": len({r["cell"] for r in rows})}}
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
