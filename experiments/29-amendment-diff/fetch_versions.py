"""Fetch the two versions of 用戶用電設備裝置規則 and the amendment's comparison table.

    uv run python experiments/29-amendment-diff/fetch_versions.py

  old   the 110-03-17 version from the database's history page -> corpus/old-1100317.txt
        (laws.law_text; the raw HTML goes to data/29-amendment-diff/pages/)
  new   the 114-05-09 version: 01's corpus/law.pdf through rag_lab.pdf.extract_text (what 27/28
        indexed) -> corpus/new-1140509.txt, stored once so the signature does not depend on the
        pypdf version at run time
  compare  the 114-05-09 修正條文對照表 (PDF, 1,081 pages, 15.7 MB) and 立法總說明 -> data/
           (git-ignored; the PDF is over the 10 MB limit). compare_table.py cuts it into rows.

corpus/manifest.json records every URL, the fetch date, sha256 of every file, the pypdf version
and the article counts seen on the pages.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import urllib.request
from datetime import date
from importlib.metadata import version
from pathlib import Path

from rag_lab.log import setup_logging
from rag_lab.pdf import extract_text

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import laws  # noqa: E402

logger = logging.getLogger(__name__)

ROOT = HERE.parents[1]
DATA = ROOT / "data/29-amendment-diff"
CORPUS = HERE / "corpus"
NEW_PDF = ROOT / "experiments/01-chunking/corpus/law.pdf"
PCODE = "J0030018"
OLD_DATE, NEW_DATE = "20210317", "20250509"  # 110-03-17, 114-05-09
URLS = {
    "old": f"https://law.moj.gov.tw/LawClass/LawOldVer.aspx?pcode={PCODE}&lnndate={OLD_DATE}&lser=001",
    "history": f"https://law.moj.gov.tw/LawClass/LawHistory.aspx?pcode={PCODE}",
    "compare": (
        "https://law.moj.gov.tw/LawClass/LawGetFile.ashx?FileId=0000395914&lan=C&type=1"
        f"&date={NEW_DATE}"
    ),
    "desc": f"https://law.moj.gov.tw/LawClass/LawGetDesc.ashx?FileId={PCODE}&date={NEW_DATE}&lan=C",
}
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
OLD_TXT = CORPUS / "old-1100317.txt"
NEW_TXT = CORPUS / "new-1140509.txt"
COMPARE_PDF = DATA / "compare-1140509.pdf"
DESC_HTML = DATA / "desc-1140509.html"


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    setup_logging("rag_lab")
    CORPUS.mkdir(exist_ok=True)
    (DATA / "pages").mkdir(parents=True, exist_ok=True)

    old_page = get(URLS["old"]).decode("utf-8")
    (DATA / "pages" / f"old-{OLD_DATE}.html").write_text(old_page, "utf-8")
    old_text = laws.law_text(old_page)
    OLD_TXT.write_text(old_text + "\n", "utf-8")
    old_name, old_amended = laws.page_meta(old_page)
    old_n = len(laws.parse_law(old_text))
    logger.info(
        "old version",
        extra={"ctx": {"name": old_name, "amended": old_amended, "articles": old_n}},
    )

    new_text = extract_text(NEW_PDF)
    NEW_TXT.write_text(new_text + "\n", "utf-8")
    new_n = len(laws.parse_law(new_text))
    logger.info("new version", extra={"ctx": {"articles": new_n, "chars": len(new_text)}})

    if not COMPARE_PDF.exists():
        COMPARE_PDF.write_bytes(get(URLS["compare"]))
    if not DESC_HTML.exists():
        DESC_HTML.write_bytes(get(URLS["desc"]))

    manifest = {
        "law": {"pcode": PCODE, "name": laws.BASE_NAME},
        "fetched": date.today().isoformat(),
        "old": {
            "amended": old_amended,
            "name_on_page": old_name,
            "url": URLS["old"],
            "articles_on_page": laws.articles_on_page(old_page),
            "articles_parsed": old_n,
            "chars": len(old_text),
            "sha256": sha256(OLD_TXT),
        },
        "new": {
            "amended": "民國 114 年 05 月 09 日",
            "source": str(NEW_PDF.relative_to(ROOT)),
            "pdf_sha256": sha256(NEW_PDF),
            "pypdf": version("pypdf"),
            "articles_parsed": new_n,
            "chars": len(new_text),
            "sha256": sha256(NEW_TXT),
        },
        "compare": {
            "url": URLS["compare"],
            "file": str(COMPARE_PDF.relative_to(ROOT)),
            "bytes": COMPARE_PDF.stat().st_size,
            "sha256": sha256(COMPARE_PDF),
        },
        "desc": {
            "url": URLS["desc"],
            "file": str(DESC_HTML.relative_to(ROOT)),
            "sha256": sha256(DESC_HTML),
        },
        "history_url": URLS["history"],
    }
    (CORPUS / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
