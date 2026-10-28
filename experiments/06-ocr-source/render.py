"""Render every page of 03's 17 table PDFs to data/06-ocr/png/<表號>-p<N>.png at 300 dpi
(pypdfium2 through pdfplumber, antialiased). data/ is not in git; re-run on the other machine.

    uv run python experiments/06-ocr-source/render.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pdfplumber

from rag_lab.log import setup_logging

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
ATTACH = ROOT / "data/01-chunking/attachments"
PNG = ROOT / "data/06-ocr/png"
sys.path.insert(0, str(HERE.parent / "04-table-lookup"))
from build_corpus import TABLES  # noqa: E402  (表號 -> FileId, the same 17 tables)

logger = logging.getLogger(__name__)
DPI = 300


def main() -> int:
    setup_logging("rag_lab")
    PNG.mkdir(parents=True, exist_ok=True)
    n = 0
    for tid, fid in TABLES.items():
        with pdfplumber.open(ATTACH / f"{fid}.pdf") as pdf:
            for i, page in enumerate(pdf.pages):
                page.to_image(resolution=DPI, antialias=True).original.save(PNG / f"{tid}-p{i}.png")
                n += 1
    logger.info("rendered", extra={"ctx": {"pages": n, "dpi": DPI}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
