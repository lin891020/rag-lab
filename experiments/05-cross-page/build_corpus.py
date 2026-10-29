"""The three tables that continue over pages -> corpus/merged/<表號>.txt (rule-merged, whole
table) and corpus/pages/<表號>-p<N>.txt (each page read on its own, first row as header on the
continuation pages). run.py reads only these, so it needs no PDFs.

    uv run python experiments/05-cross-page/build_corpus.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from rag_lab.log import setup_logging

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
ATTACH = ROOT / "data/01-chunking/attachments"
CORPUS = HERE / "corpus"
sys.path.insert(0, str(HERE))
from tables import load_page, load_pages, load_table, to_markdown  # noqa: E402

logger = logging.getLogger(__name__)

TABLES: dict[str, str] = {
    "表五六": "0000395929",
    "表五八七～一": "0000395996",
    "表五八七～三": "0000395998",
}
# 03's header rule (last of the first four rows with a colspan) misreads the two 危險場所 tables:
# their first column is a two-level label whose spans sit in every row. One header row each.
HEADER_ROWS: dict[str, int] = {
    "表五八七～一": 1,
    "表五八七～三": 1,
    "表五六": 2,
}  # 表五六: 2 since the 2026-10-29 audit (page 0 fell back to 1, merged guessed 2)


def main() -> int:
    setup_logging("rag_lab")
    stats: dict[str, dict] = {}
    for tid, fid in TABLES.items():
        pdf = ATTACH / f"{fid}.pdf"
        hr = HEADER_ROWS.get(tid)
        merged = load_table(pdf, header_rows=hr)
        out = CORPUS / "merged" / f"{tid}.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(to_markdown(merged), encoding="utf-8")
        pages = {}
        for g in load_pages(pdf):
            # page 0 keeps the table's own header depth; a continuation page has no header,
            # so the naive reading promotes its first row (load_page's default)
            t = load_page(pdf, g.page_no, header_rows=hr if g.page_no == 0 else None)
            p = CORPUS / "pages" / f"{tid}-p{g.page_no}.txt"
            p.parent.mkdir(parents=True, exist_ok=True)
            text = to_markdown(t)
            p.write_text(text, encoding="utf-8")
            pages[g.page_no] = {"chars": len(text), "rows": len(t.data_rows())}
        stats[tid] = {
            "pages_merged": merged.pages,
            "merged_chars": len(to_markdown(merged)),
            "merged_rows": len(merged.data_rows()),
            "pages": pages,
        }
    (CORPUS / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), "utf-8")
    logger.info("corpus built", extra={"ctx": {"tables": len(stats)}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
