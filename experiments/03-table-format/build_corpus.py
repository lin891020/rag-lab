"""Render the 17 tables the questions use into corpus/{markdown,html,kv}/<表號>.txt.

    uv run python experiments/03-table-format/build_corpus.py

Source PDFs are 01's attachments (data/01-chunking/attachments, git-ignored; re-fetch with
experiments/01-chunking/fetch_corpus.py). The flat baseline is 01's corpus/tables/*.txt and is
not copied. Outputs are committed: they are small and the experiment must be re-runnable on the
other machine without the PDFs.
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
from tables import load_table, to_html, to_kv, to_markdown  # noqa: E402

logger = logging.getLogger(__name__)

# 表號 -> attachment FileId (from the law page; see 01/fetch_corpus.py)
TABLES: dict[str, str] = {
    "表二五～一": "0000395917",
    "表二五～二": "0000395918",
    "表二五～四": "0000395920",
    "表二五～五": "0000395921",
    "表二五～六": "0000395922",
    "表二五～七": "0000395923",
    "表三六": "0000395924",
    "表四○": "0000395925",
    "表五二": "0000395926",
    "表五五": "0000395928",
    "表五六": "0000395929",
    "表六一": "0000395932",
    "表八六～三": "0000395935",
    "表八八～二": "0000395937",
    "表九二": "0000395938",
    "表九三～一": "0000395939",
    "表九三～二": "0000395940",
}
# header depth the geometry rule gets wrong (a human decision, recorded in the README)
HEADER_ROWS: dict[str, int] = {"表四○": 1}
RENDER = {"markdown": to_markdown, "html": to_html, "kv": to_kv}


def main() -> int:
    setup_logging("rag_lab")
    stats = {}
    for tid, fid in TABLES.items():
        t = load_table(ATTACH / f"{fid}.pdf", header_rows=HEADER_ROWS.get(tid))
        stats[tid] = {"header_rows": t.header_rows, "rows": len(t.data_rows()), "cols": t.n_cols()}
        for fmt, fn in RENDER.items():
            out = CORPUS / fmt / f"{tid}.txt"
            out.parent.mkdir(parents=True, exist_ok=True)
            text = fn(t)
            out.write_text(text, encoding="utf-8")
            stats[tid][fmt] = len(text)
    (CORPUS / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), "utf-8")
    logger.info("corpus built", extra={"ctx": {"tables": len(stats)}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
