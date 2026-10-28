"""17 tables -> corpus/tables.json (flattened grids, so run.py needs no PDFs) and
corpus/{md,md-range}/<表號>.txt for reading. The row-selection conditions are rendered at run
time from tables.json.

    uv run python experiments/04-table-lookup/build_corpus.py
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
from tables import Table, expand_ranges, load_table, table_to_json, to_markdown  # noqa: E402

logger = logging.getLogger(__name__)

TABLES: dict[str, str] = {  # same 17 as 03
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
HEADER_ROWS: dict[str, int] = {"表四○": 1}


def main() -> int:
    setup_logging("rag_lab")
    blob: dict[str, dict] = {}
    for tid, fid in TABLES.items():
        t: Table = load_table(ATTACH / f"{fid}.pdf", header_rows=HEADER_ROWS.get(tid))
        blob[tid] = table_to_json(t)
        for fmt, text in (("md", to_markdown(t)), ("md-range", to_markdown(expand_ranges(t)))):
            out = CORPUS / fmt / f"{tid}.txt"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
    (CORPUS / "tables.json").write_text(json.dumps(blob, ensure_ascii=False, indent=1), "utf-8")
    logger.info("corpus built", extra={"ctx": {"tables": len(blob)}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
