"""Fetch the five added laws and their table attachments from the national law database.

    uv run python experiments/28-hosted-bigger-corpus/fetch_corpus.py

Each law's print page -> corpus/laws/<pcode>.txt (laws.law_text) and a manifest entry (name,
修正日期, articles counted on the raw page, attachments). Table attachments (表 links, PDF) are
downloaded to data/28-hosted-bigger-corpus/attachments/<pcode>/ (git-ignored) and converted with
01's `table_to_text` -> corpus/tables/<pcode>/<表號>.txt. 圖 links are dropped, as in 01.
The print pages go to data/ too, so a later fetch can be compared with this one. Tables that
the page offers only as .ODT or .DOC are not fetched (輸配電 表三三○／三三七／三三八), and two of
職安's PDFs hold only their titles as text (表一, 表二): kept as they come, written in the README.
"""

from __future__ import annotations

import html
import json
import logging
import re
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

from rag_lab.log import setup_logging
from rag_lab.pdf import table_to_text

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import laws  # noqa: E402

logger = logging.getLogger(__name__)

ROOT = HERE.parents[1]
DATA = ROOT / "data/28-hosted-bigger-corpus"
PRINT = "https://law.moj.gov.tw/LawClass/LawAll.aspx?media=print&pcode={pcode}"
PAGE = "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode={pcode}"
FILE_URL = "https://law.moj.gov.tw/LawClass/LawGetFile.ashx?FileId={fid}&lan=C"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
_LINK = re.compile(r'LawGetFile\.ashx\?FileId=(\d+)&(?:amp;)?lan=C"[^>]*>([^<]+?)\.PDF')


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def parse_attachments(page_html: str) -> list[tuple[str, str, str]]:
    """01's: (file_id, table_id, title) for every 表 link; 圖 links are dropped."""
    out = []
    for fid, name in _LINK.findall(page_html):
        name = html.unescape(name).strip()
        if not name.startswith("表"):
            continue
        out.append((fid, re.split(r"[：:\s]", name, maxsplit=1)[0], name))
    return out


def main() -> int:
    setup_logging("rag_lab")
    laws.LAWS_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    for pcode, want in laws.LAWS.items():
        page = get(PRINT.format(pcode=pcode)).decode("utf-8")
        full = get(PAGE.format(pcode=pcode)).decode("utf-8")
        (DATA / "pages").mkdir(parents=True, exist_ok=True)
        (DATA / "pages" / f"{pcode}.html").write_text(page, "utf-8")
        name, amended = laws.page_meta(page)
        if name != want:
            raise SystemExit(f"{pcode} is {name!r}, expected {want!r}")
        text = laws.law_text(page)
        (laws.LAWS_DIR / f"{pcode}.txt").write_text(text + "\n", "utf-8")
        items = parse_attachments(full)
        folder = laws.TABLES_DIR / pcode
        att = DATA / "attachments" / pcode
        if items:
            folder.mkdir(parents=True, exist_ok=True)
            att.mkdir(parents=True, exist_ok=True)
        for fid, tid, title in items:
            pdf = att / f"{fid}.pdf"
            if not pdf.exists():
                pdf.write_bytes(get(FILE_URL.format(fid=fid)))
                time.sleep(0.5)
            t = table_to_text(pdf)
            if not t.startswith(tid):
                logger.warning(
                    "title mismatch", extra={"ctx": {"pcode": pcode, "fid": fid, "want": title}}
                )
            (folder / f"{tid}.txt").write_text(t + "\n", "utf-8")
        manifest[pcode] = {
            "name": name,
            "amended": amended,
            "url": PAGE.format(pcode=pcode),
            "fetched": date.today().isoformat(),
            "articles_on_page": laws.articles_on_page(page),
            "chars": len(text),
            "tables": [{"file_id": f, "table": t, "title": n} for f, t, n in items],
        }
        logger.info(
            "law fetched",
            extra={"ctx": {"pcode": pcode, "chars": len(text), "tables": len(items)}},
        )
    (laws.LAWS_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
