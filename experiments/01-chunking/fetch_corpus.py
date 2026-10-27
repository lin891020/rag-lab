"""Fetch the 109 table attachments of 用戶用電設備裝置規則 and convert each to plain text.

    uv run python experiments/01-chunking/fetch_corpus.py

Raw PDFs land in data/01-chunking/attachments/ (git-ignored); text in corpus/tables/ (committed).
law.pdf itself is committed; this script only checks its sha256 against README.
"""

from __future__ import annotations

import hashlib
import html
import logging
import re
import sys
import urllib.request
from pathlib import Path

from rag_lab.log import setup_logging
from rag_lab.pdf import table_to_text

logger = logging.getLogger(__name__)

PCODE = "J0030018"
LAW_PAGE = f"https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode={PCODE}"
FILE_URL = "https://law.moj.gov.tw/LawClass/LawGetFile.ashx?FileId={fid}&lan=C"
HERE = Path(__file__).parent
ROOT = HERE.parents[1]
RAW_DIR = ROOT / "data/01-chunking/attachments"
TABLES_DIR = HERE / "corpus/tables"

_LINK = re.compile(r'LawGetFile\.ashx\?FileId=(\d+)&(?:amp;)?lan=C"[^>]*>([^<]+?)\.PDF')


def parse_attachments(page_html: str) -> list[tuple[str, str, str]]:
    """(file_id, table_id, title) for every 表 link; 圖 links are dropped."""
    out = []
    for fid, name in _LINK.findall(page_html):
        name = html.unescape(name).strip()
        if not name.startswith("表"):
            continue
        out.append((fid, name.split()[0], name))
    return out


def main() -> int:
    setup_logging("rag_lab")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    page = urllib.request.urlopen(LAW_PAGE, timeout=30).read().decode("utf-8")
    items = parse_attachments(page)
    logger.info("attachments listed", extra={"ctx": {"tables": len(items)}})
    for fid, table_id, title in items:
        pdf = RAW_DIR / f"{fid}.pdf"
        if not pdf.exists():
            urllib.request.urlretrieve(FILE_URL.format(fid=fid), pdf)
        text = table_to_text(pdf)
        if not text.startswith(table_id):
            logger.warning("title mismatch", extra={"ctx": {"fid": fid, "want": title, "got": text[:40]}})
        (TABLES_DIR / f"{table_id}.txt").write_text(text, encoding="utf-8")
    sha = hashlib.sha256((HERE / "corpus/law.pdf").read_bytes()).hexdigest()
    logger.info("done", extra={"ctx": {"tables_written": len(items), "law_sha256": sha}})
    print(f"law.pdf sha256 {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
