"""Thirty naturally messy regulation amendments from the government gazettes.

    uv run python experiments/09-messy-real/build_corpus.py [--seed 0] [--max-scan 300]

Source: the National Central Library's gazette open data (gov.xml, one half-year snapshot,
every record carries a PDF link). Kept: records whose caption says 修正 and names a 條文／規則／
辦法／標準, with a file, 3–30 pages. Two strata of PER_TYPE each: 預告…草案 (draft: the
three-column 修正條文／現行條文／說明 table) and 令：修正 (final: the amended articles as text).
Minima inside the 30: local gazettes, each PDF producer family, a PDF with scanned (textless)
pages and one whose text layer is (cid:N) garbage — the messes 08 said it could detect, plus
the one it could not. The pick scans the shuffled pool, downloads
a candidate, measures it (producer from the PDF metadata, health from doc_health) and keeps it
while it still helps a quota; it stops at --max-scan downloads and reports what stayed unmet.

Outputs (in git): corpus/pdf/<sysid>.pdf, corpus/manifest.json, corpus/health.csv.
Cache (not in git): data/09-messy-real/gov.xml, data/09-messy-real/pdf-cache/.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import shutil
import sys
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pymupdf

from rag_lab.log import setup_logging

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
DATA = ROOT / "data/09-messy-real"
CORPUS = HERE / "corpus"
sys.path.insert(0, str(HERE))
from doc_health import inspect_pdf  # noqa: E402

logger = logging.getLogger(__name__)

XML_URL = "https://gaz.ncl.edu.tw/opendata/gov.xml"
PER_TYPE = 15
MINIMA = {"local": 6, "producer": 2, "scanned": 1, "garbage": 1}
FAMILIES = ("itext", "ilovepdf", "word", "distiller")
MIN_PAGES, MAX_PAGES = 3, 30
CENTRAL = ("行政院公報", "總統府公報", "立法院公報", "司法院公報", "考試院公報", "監察院公報")
_KIND = ("條文", "規則", "辦法", "標準")
_REC = re.compile(r"<Record>(.*?)</Record>", re.S)
_PAGES = re.compile(r"頁(\d+)-(\d+)")


def _field(rec: str, tag: str) -> str:
    m = re.search(rf"<{tag}><!\[CDATA\[(.*?)\]\]", rec)
    return m.group(1).strip() if m else ""


def parse_records(xml: str) -> list[dict]:
    out = []
    for rec in _REC.findall(xml):
        m = _PAGES.search(_field(rec, "Pages"))
        out.append(
            {
                "sysid": _field(rec, "Sysid"),
                "caption": _field(rec, "Caption"),
                "gazette": _field(rec, "GazName"),
                "date": _field(rec, "PubDate"),
                "pages": int(m.group(2)) - int(m.group(1)) + 1 if m else 0,
                "file": _field(rec, "EFile"),
            }
        )
    logger.info("records parsed", extra={"ctx": {"n": len(out)}})
    return out


def wanted(r: dict) -> bool:
    cap = r["caption"]
    return bool(
        r["file"]
        and "修正" in cap
        and any(k in cap for k in _KIND)
        and MIN_PAGES <= r["pages"] <= MAX_PAGES
    )


def doc_type(r: dict) -> str:
    return "draft" if ("預告" in r["caption"] or "草案" in r["caption"]) else "final"


def is_local(r: dict) -> bool:
    return not any(r["gazette"].startswith(c) for c in CENTRAL)


def producer_family(producer: str) -> str:
    p = producer.lower()
    if "itext" in p:
        return "itext"
    if "ilovepdf" in p:
        return "ilovepdf"
    if "word" in p or "microsoft" in p:
        return "word"
    if "distiller" in p:
        return "distiller"
    return "other"


def _unmet(chosen: list[dict]) -> dict[str, int]:
    """quota name -> how many more are needed."""
    need = {}
    for t in ("draft", "final"):
        need[t] = PER_TYPE - sum(c["type"] == t for c in chosen)
    need["local"] = MINIMA["local"] - sum(c["local"] for c in chosen)
    need["scanned"] = MINIMA["scanned"] - sum(c["scanned"] for c in chosen)
    need["garbage"] = MINIMA["garbage"] - sum(c["garbage"] for c in chosen)
    for fam in FAMILIES:
        need[fam] = MINIMA["producer"] - sum(c["producer"] == fam for c in chosen)
    return {k: v for k, v in need.items() if v > 0}


def pick(
    pool: list[dict],
    *,
    features: Callable[[dict], dict | None],
    seed: int,
    max_scan: int = 300,
) -> tuple[list[dict], int]:
    """Scan the shuffled pool; keep a candidate while its type still has room and either it
    closes an unmet minimum or enough slots of its type remain after reserving one per unmet
    minimum count. Candidates turned away only by that reserve are kept as spares and fill the
    remaining slots once the scan ends (pool exhausted, max_scan reached, or all quotas met), so
    an unsatisfiable minimum is reported, not looped on. Returns (chosen, candidates examined)."""
    rng = random.Random(seed)
    order = pool[:]
    rng.shuffle(order)
    chosen: list[dict] = []
    spare: list[dict] = []
    scanned = 0
    for r in order:
        need = _unmet(chosen)
        if not need or scanned >= max_scan:
            break
        f = features(r)
        scanned += 1
        if f is None:
            continue
        cand = {**r, **f}
        if need.get(cand["type"], 0) <= 0:
            continue
        helps = (
            (cand["local"] and "local" in need)
            or (cand["scanned"] and "scanned" in need)
            or (cand["garbage"] and "garbage" in need)
            or cand["producer"] in need
        )
        reserve = sum(v for k, v in need.items() if k not in ("draft", "final"))
        if helps or need[cand["type"]] > reserve:
            chosen.append(cand)
        else:
            spare.append(cand)
    for cand in spare:
        if _unmet(chosen).get(cand["type"], 0) > 0:
            chosen.append(cand)
    unmet = _unmet(chosen)
    if unmet:
        logger.warning("quotas unmet", extra={"ctx": {"unmet": unmet, "scanned": scanned}})
    logger.info("picked", extra={"ctx": {"n": len(chosen), "scanned": scanned}})
    return chosen, scanned


# -- the real features: download, read the producer, run the health check ----------------------


def download(url: str, dst: Path, *, retries: int = 3) -> bool:
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            if data[:4] != b"%PDF":
                logger.warning("not a pdf", extra={"ctx": {"url": url, "bytes": len(data)}})
                return False
            dst.write_bytes(data)
            return True
        except (OSError, ValueError) as exc:
            logger.warning("download failed", extra={"ctx": {"url": url, "try": i}}, exc_info=exc)
            time.sleep(2)
    return False


def measure(r: dict, cache: Path) -> dict | None:
    pdf = cache / f"{r['sysid']}.pdf"
    if not pdf.exists() and not download(r["file"], pdf):
        return None
    try:
        with pymupdf.open(pdf) as doc:
            producer = doc.metadata.get("producer") or ""
    except (RuntimeError, ValueError) as exc:
        logger.warning("unreadable pdf", extra={"ctx": {"sysid": r["sysid"]}}, exc_info=exc)
        return None
    h = inspect_pdf(pdf)
    return {
        "type": doc_type(r),
        "local": is_local(r),
        "producer": producer_family(producer),
        "producer_raw": producer,
        "scanned": h["text_pages"] < h["pages"],
        "garbage": h["garbage_pages"] > 0,
        "rotated": h["rotated_pages"] > 0,
        "health": h,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-scan", type=int, default=300)
    args = ap.parse_args(argv)
    setup_logging("rag_lab")
    DATA.mkdir(parents=True, exist_ok=True)
    cache = DATA / "pdf-cache"
    cache.mkdir(exist_ok=True)
    xml = DATA / "gov.xml"
    if not xml.exists():
        logger.info("downloading gov.xml", extra={"ctx": {"url": XML_URL}})
        req = urllib.request.Request(XML_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            xml.write_bytes(resp.read())
    pool = [r for r in parse_records(xml.read_text(encoding="utf-8")) if wanted(r)]
    logger.info("pool", extra={"ctx": {"n": len(pool)}})
    chosen, scanned = pick(
        pool, features=lambda r: measure(r, cache), seed=args.seed, max_scan=args.max_scan
    )
    (CORPUS / "pdf").mkdir(parents=True, exist_ok=True)
    for c in chosen:
        shutil.copy(cache / f"{c['sysid']}.pdf", CORPUS / "pdf" / f"{c['sysid']}.pdf")
    chosen.sort(key=lambda c: (c["type"], c["sysid"]))
    manifest = {
        "source": XML_URL,
        "snapshot_dates": [min(c["date"] for c in chosen), max(c["date"] for c in chosen)],
        "seed": args.seed,
        "pool": len(pool),
        "scanned": scanned,
        "per_type": PER_TYPE,
        "minima": MINIMA,
        "unmet": _unmet(chosen),
        "docs": chosen,
    }
    (CORPUS / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    with (CORPUS / "health.csv").open("w", newline="", encoding="utf-8") as f:
        rows = [c["health"] for c in chosen]
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    logger.info(
        "corpus built",
        extra={
            "ctx": {
                "docs": len(chosen),
                "draft": sum(c["type"] == "draft" for c in chosen),
                "local": sum(c["local"] for c in chosen),
                "scanned": sum(c["scanned"] for c in chosen),
                "garbage": sum(c["garbage"] for c in chosen),
                "producers": {f: sum(c["producer"] == f for c in chosen) for f in FAMILIES},
            }
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
