"""The native (non-PDF) copy of each gazette notice: the ministry law-system page and its
attachments, or the national law database page for laws and orders.

    uv run python experiments/14-native-source/fetch_native.py

DOCS maps each 13 / 09 sysid to where its regulation lives (found 2026-10-29 with the shared
systems' GET search `LawResult.aspx?NLawTypeID=all&KW=…&name=1&content=1`, or the national
database search). Two of 13's central notices have no reachable system (僑務委員會: none
found; 交通部 motclaw.motc.gov.tw: certificate error) and are left out.

Outputs (in git): corpus/native/<sysid>.html, corpus/native/<sysid>-att-N.<ext>,
corpus/manifest.json (caption / type / kind / health copied from the source experiment, plus
the system URL, its 修正日期, whether it matches the gazette date, and the attachment list).
"""

from __future__ import annotations

import html as htmllib
import json
import logging
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from rag_lab.log import setup_logging

HERE = Path(__file__).parent
CORPUS = HERE / "corpus"
NATIVE = CORPUS / "native"
MANIFESTS = {
    "13": HERE.parent / "13-points-tables" / "corpus" / "manifest.json",
    "09": HERE.parent / "09-messy-real" / "corpus" / "manifest.json",
}
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
logger = logging.getLogger(__name__)

# sysid -> (source experiment, system base, law id)
DOCS: dict[str, tuple[str, str, str]] = {
    "E2418689": ("13", "https://edu.law.moe.gov.tw", "FL038754"),
    "E2418690": ("13", "https://edu.law.moe.gov.tw", "FL041432"),
    "E2421778": ("13", "https://edu.law.moe.gov.tw", "FL046130"),
    "E2421751": ("13", "https://law.moa.gov.tw", "GL001104"),
    "E2423169": ("13", "https://law.moa.gov.tw", "GL001535"),
    "E2428884": ("13", "https://law.moa.gov.tw", "GL001940"),
    "E2423850": ("13", "https://law-out.mof.gov.tw", "FL047059"),
    "E2425151": ("13", "https://law.fsc.gov.tw", "GL003991"),
    "E2416980": ("13", "https://law.moea.gov.tw", "GL000402"),
    "E2420688": ("09", "https://law.moa.gov.tw", "FL032062"),
    "E2418038": ("09", "https://glrs.moi.gov.tw", "FL005037"),
    "E2426119": ("09", "https://law.moea.gov.tw", "FL011232"),
    "E2427268": ("09", "https://law.moea.gov.tw", "FL011214"),
    "E2417271": ("09", "MOJ", "G0390053"),
    "E2419634": ("09", "MOJ", "D0060076"),
    "E2419637": ("09", "MOJ", "J0140001"),
}
_DATE = re.compile(
    r"(?:修正日期|最新異動日期|發布日期|公發布日|公布日期)[：:\s]*民國\s*(\d+)\s*年\s*(\d+)\s*月\s*(\d+)\s*日"
)
_REVISED = re.compile(r"修正日期[：:\s]*民國\s*(\d+)\s*年\s*(\d+)\s*月\s*(\d+)\s*日")
# notices the system has since amended: the exact version is fetched from a history page instead
VERSIONS: dict[str, str] = {
    "E2417271": "https://law.moj.gov.tw/LawClass/LawOldVer.aspx?pcode=G0390053&lnndate=20240704&lser=001",
}
_ATT = re.compile(r'<a[^>]+href="((?:\./)?Download\.ashx\?FileID=\d+[^"]*)"[^>]*>(.*?)</a>', re.S)


def page_url(base: str, law_id: str) -> str:
    if base == "MOJ":
        return f"https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode={law_id}"
    return f"{base}/LawContent.aspx?id={law_id}"


def fetch(url: str) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90).read()


def revised_date(page: str) -> str | None:
    """民國 date on the page as YYYYMMDD (西元)."""
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", htmllib.unescape(page)))
    m = _REVISED.search(text) or _DATE.search(text)
    if not m:
        return None
    y, mo, d = (int(x) for x in m.groups())
    return f"{y + 1911:04d}{mo:02d}{d:02d}"


def attachments(page: str, base: str) -> list[dict]:
    out = []
    for m in _ATT.finditer(page):
        name = re.sub(r"\s+", "", re.sub(r"<[^>]+>", "", htmllib.unescape(m.group(2))))
        href = htmllib.unescape(m.group(1)).lstrip("./")
        if name and "type=LAW" in href:
            out.append({"name": name, "url": f"{base}/{href}"})
    seen: set[str] = set()
    uniq = []
    for a in out:
        if a["url"] not in seen:
            seen.add(a["url"])
            uniq.append(a)
    return uniq


def main() -> int:
    setup_logging("rag_lab")
    NATIVE.mkdir(parents=True, exist_ok=True)
    sources = {
        k: {d["sysid"]: d for d in json.loads(p.read_text("utf-8"))["docs"]}
        for k, p in MANIFESTS.items()
    }
    docs = []
    for sysid, (src, base, law_id) in DOCS.items():
        s = sources[src][sysid]
        url = VERSIONS.get(sysid) or page_url(base, law_id)
        raw = fetch(url)
        page = raw.decode("utf-8", "ignore")
        (NATIVE / f"{sysid}.html").write_bytes(raw)
        date = revised_date(page)
        atts = []
        for i, a in enumerate(attachments(page, base) if base != "MOJ" else [], 1):
            ext = Path(urllib.parse.unquote(a["name"])).suffix.lower() or ".bin"
            fname = f"{sysid}-att-{i}{ext}"
            try:
                (NATIVE / fname).write_bytes(fetch(a["url"]))
                atts.append({**a, "file": fname, "ext": ext})
            except OSError as exc:
                logger.warning("attachment failed", extra={"ctx": {"url": a["url"]}}, exc_info=exc)
            time.sleep(0.5)
        docs.append(
            {
                "sysid": sysid,
                "from": src,
                "caption": s["caption"],
                "gazette": s["gazette"],
                "date": s["date"],
                "pages": s["pages"],
                "type": s.get("type"),
                "kind": "points"
                if s.get("type") == "points"
                else ("tables" if s.get("type") == "tables" else "law"),
                "local": s.get("local", False),
                "health": s.get("health"),
                "system": base,
                "law_id": law_id,
                "url": url,
                "revised": date,
                "same_version": date == s["date"],
                "history_page": sysid in VERSIONS,
                "attachments": atts,
            }
        )
        logger.info(
            "fetched",
            extra={
                "ctx": {"sysid": sysid, "revised": date, "gazette": s["date"], "att": len(atts)}
            },
        )
        time.sleep(1)
    manifest = {"fetched": time.strftime("%Y-%m-%d"), "docs": docs}
    (CORPUS / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), "utf-8"
    )
    logger.info(
        "manifest",
        extra={
            "ctx": {
                "docs": len(docs),
                "same_version": sum(d["same_version"] for d in docs),
                "attachments": sum(len(d["attachments"]) for d in docs),
            }
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
