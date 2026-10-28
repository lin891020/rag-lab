"""Build every messy variant and extract it exactly the way the clean pipeline would.

    uv run python experiments/08-messy/build_corpus.py [--text] [--tables] [--levels phone,copy3] [--from-raw]

Text track (01's regulation, 283 pages)         -> corpus/text/<variant>[.<extractor>].txt
  clean           01's law.pdf                                  pypdf
  header-footer   title / date / 第 N 頁 stamped on every page     pypdf
  two-column      the same lines re-typeset in two columns        pypdf AND pdfplumber
  concat          law.pdf + 輸配電設備裝置規則 (J0030019) typeset from the MOJ page   pypdf
Table track (03's 17 tables)                    -> corpus/tables/<route>/<表號>.txt
  clean           04's markdown (pdfplumber geometry)
  scan-clean      06's glm2 markdown (300 dpi page image -> glm-ocr, HTML prompt)
  scan-phone      the same PNG degraded (rotate, blur, shadow, jpeg, 200 dpi) -> glm-ocr
  scan-copy3      + salt-and-pepper, binarised, 150 dpi -> glm-ocr
  borderless-text 04's grid typeset with no lines -> pdfplumber text strategy -> 03's pipeline
  borderless-flat the same PDF -> rag_lab.pdf.table_to_text (what happens when no table is found)
PDFs and PNGs go to data/08-messy/ (not in git); texts and stats.json go to corpus/ (in git).
"""

from __future__ import annotations

import argparse
import hashlib
import html as html_mod
import json
import logging
import re
import sys
import time
import urllib.request
from pathlib import Path

import pdfplumber
import pymupdf
from pypdf import PdfReader

from rag_lab.log import setup_logging
from rag_lab.pdf import extract_text, table_to_text

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
ONE = HERE.parent / "01-chunking"
FOUR = HERE.parent / "04-table-lookup"
SIX = HERE.parent / "06-ocr-source"
DATA = ROOT / "data/08-messy"
CORPUS = HERE / "corpus"
PNG6 = ROOT / "data/06-ocr/png"
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(SIX))
from degrade import (  # noqa: E402
    A4,
    borderless_table,
    concat,
    degrade_png,
    header_footer,
    two_column,
    typeset,
)
from labels import line_map_from_text  # noqa: E402
from ocr import GLM_PROMPT_NL, glm_ocr, glm_to_markdown  # noqa: E402
from tables import load_table, to_markdown  # noqa: E402

logger = logging.getLogger(__name__)

TITLE = "用戶用電設備裝置規則"
DISTRACTOR_PCODE = "J0030019"  # 輸配電設備裝置規則: same vocabulary, different subject
DISTRACTOR_URL = f"https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode={DISTRACTOR_PCODE}"
TEXT_STRATEGY = {"vertical_strategy": "text", "horizontal_strategy": "text"}
LEVELS = ("phone", "copy3")

# -- the distractor regulation from the MOJ page ----------------------------------------------

_CHAP = re.compile(r'<div class="h3 char-2">(.*?)</div>')
_SECT = re.compile(r'<div class="h3 char-3">(.*?)</div>')
_ART = re.compile(r'<div class="col-no">\s*<a[^>]*>(第 \d+ 條)</a>')
_LINE = re.compile(r'<div class="(line-\d+)( show-number)?">(.*?)</div>')
_TAG = re.compile(r"<[^>]+>")


def moj_lines(html: str) -> list[str]:
    """Lines shaped like law.pdf's text layer: chapter / section lines with their indent,
    `第 N 條` on its own line, numbered 項 as '1   …' when an article has several, 款 as-is."""
    events: list[tuple[int, str, str, str]] = []
    for m in _CHAP.finditer(html):
        events.append((m.start(), "chap", m.group(1), ""))
    for m in _SECT.finditer(html):
        events.append((m.start(), "sect", m.group(1), ""))
    for m in _ART.finditer(html):
        events.append((m.start(), "art", m.group(1), ""))
    for m in _LINE.finditer(html):
        events.append(
            (
                m.start(),
                "line",
                html_mod.unescape(_TAG.sub("", m.group(3))).strip(),
                m.group(2) or "",
            )
        )
    events.sort()
    out: list[str] = []
    i = 0
    while i < len(events):
        _, kind, text, flag = events[i]
        if kind in ("chap", "sect"):
            out.append(html_mod.unescape(text).rstrip())
            i += 1
        elif kind == "art":
            out.append(text)
            j = i + 1
            block: list[tuple[str, str]] = []
            while j < len(events) and events[j][1] == "line":
                block.append((events[j][2], events[j][3]))
                j += 1
            numbered = [t for t, f in block if f]
            k = 0
            for t, f in block:
                if f and len(numbered) > 1:
                    k += 1
                    out.append(f"{k}   {t}")
                else:
                    out.append(t)
            i = j
        else:
            i += 1
    return out


def fetch_distractor() -> list[str]:
    DATA.mkdir(parents=True, exist_ok=True)
    cached = DATA / f"{DISTRACTOR_PCODE}.html"
    if not cached.exists():
        req = urllib.request.Request(DISTRACTOR_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            cached.write_bytes(r.read())
        logger.info(
            "fetched", extra={"ctx": {"url": DISTRACTOR_URL, "bytes": cached.stat().st_size}}
        )
    lines = moj_lines(cached.read_text("utf-8"))
    (CORPUS / "distractor.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


# -- text track ----------------------------------------------------------------------------------


def _plumber_text(pdf: Path) -> str:
    with pdfplumber.open(pdf) as doc:
        return "\n".join((p.extract_text() or "") for p in doc.pages)


def build_text() -> dict:
    pdf_dir = DATA / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    out_dir = CORPUS / "text"
    out_dir.mkdir(parents=True, exist_ok=True)
    law = ONE / "corpus/law.pdf"
    stats: dict = {"law_pdf_sha256": hashlib.sha256(law.read_bytes()).hexdigest()}

    clean = extract_text(law)
    (out_dir / "clean.txt").write_text(clean, encoding="utf-8")
    stats["clean"] = {"chars": len(clean), "pages": len(PdfReader(str(law)).pages)}

    hf = pdf_dir / "header-footer.pdf"
    stats["header-footer"] = header_footer(law, hf, title=TITLE)
    text = extract_text(hf)
    (out_dir / "header-footer.txt").write_text(text, encoding="utf-8")
    stats["header-footer"]["chars"] = len(text)

    items = line_map_from_text(clean)
    two = pdf_dir / "two-column.pdf"
    typeset_lines = two_column(items, two)
    (CORPUS / "two-column.lines.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in typeset_lines), encoding="utf-8"
    )
    stats["two-column"] = {"lines_in": len(items), "lines_out": len(typeset_lines)}
    for name, text in (("pypdf", extract_text(two)), ("plumber", _plumber_text(two))):
        (out_dir / f"two-column.{name}.txt").write_text(text, encoding="utf-8")
        stats["two-column"][f"chars_{name}"] = len(text)
    stats["two-column"]["pages"] = len(PdfReader(str(two)).pages)

    dis_lines = fetch_distractor()
    dis = pdf_dir / "distractor.pdf"
    typeset([(ln, "") for ln in dis_lines], dis, columns=1)
    cat = pdf_dir / "concat.pdf"
    stats["concat"] = concat(law, dis, cat)
    text = extract_text(cat)
    (out_dir / "concat.txt").write_text(text, encoding="utf-8")
    stats["concat"].update(chars=len(text), distractor_lines=len(dis_lines))
    return stats


# -- table track ---------------------------------------------------------------------------------


def table_ids() -> list[str]:
    return sorted(p.stem for p in (FOUR / "corpus/md").glob("*.txt"))


def _copy_route(src_dir: Path, route: str) -> dict:
    dst = CORPUS / "tables" / route
    dst.mkdir(parents=True, exist_ok=True)
    info = {}
    for tid in table_ids():
        text = (src_dir / f"{tid}.txt").read_text("utf-8")
        (dst / f"{tid}.txt").write_text(text, encoding="utf-8")
        info[tid] = {"chars": len(text)}
    return info


def build_scan(level: str, *, from_raw: bool) -> dict:
    png_dir = DATA / "png" / level
    png_dir.mkdir(parents=True, exist_ok=True)
    route = f"scan-{level}"
    dst = CORPUS / "tables" / route
    dst.mkdir(parents=True, exist_ok=True)
    raw_dir = CORPUS / "tables" / f"{route}-raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    info: dict = {}
    for tid in table_ids():
        parts = []
        seconds = 0.0
        pages = sorted(PNG6.glob(f"{tid}-p*.png"))
        for src in pages:
            png = png_dir / src.name
            params = degrade_png(src, png, level=level, seed=0)
            cached = raw_dir / f"{src.stem}.html"
            t0 = time.time()
            if from_raw and cached.exists():
                raw = cached.read_text("utf-8")
            else:
                raw = glm_ocr(png, prompt=GLM_PROMPT_NL)
                cached.write_text(raw, encoding="utf-8")
            seconds += time.time() - t0
            parts.append(glm_to_markdown(raw))
        text = "\n".join(parts)
        (dst / f"{tid}.txt").write_text(text, encoding="utf-8")
        info[tid] = {"pages": len(pages), "seconds": round(seconds, 1), "chars": len(text)}
        logger.info("ocr", extra={"ctx": {"route": route, "table": tid, "pages": len(pages)}})
    info["_params"] = {k: v for k, v in params.items() if k != "seed"} if pages else {}
    # the degraded pages packed as an image-only PDF, for doc_health.py (no text layer)
    doc = pymupdf.open()
    for png in sorted(png_dir.glob("*.png")):
        page = doc.new_page(width=A4.width, height=A4.height)
        page.insert_image(page.rect, filename=str(png))
    (DATA / "pdf").mkdir(parents=True, exist_ok=True)
    doc.save(DATA / "pdf" / f"{route}.pdf")
    doc.close()
    return info


def build_borderless() -> dict:
    gt = json.loads((FOUR / "corpus/tables.json").read_text("utf-8"))
    pdf_dir = DATA / "pdf/borderless"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    text_dir = CORPUS / "tables/borderless-text"
    flat_dir = CORPUS / "tables/borderless-flat"
    text_dir.mkdir(parents=True, exist_ok=True)
    flat_dir.mkdir(parents=True, exist_ok=True)
    info: dict = {}
    for tid in table_ids():
        pdf = pdf_dir / f"{tid}.pdf"
        params = borderless_table(gt[tid], pdf)
        with pdfplumber.open(pdf) as doc:
            ruled = sum(len(p.find_tables()) for p in doc.pages)
        t = load_table(pdf, header_rows=1, table_settings=TEXT_STRATEGY)
        md = to_markdown(t)
        (text_dir / f"{tid}.txt").write_text(md, encoding="utf-8")
        flat = table_to_text(pdf)
        (flat_dir / f"{tid}.txt").write_text(flat, encoding="utf-8")
        info[tid] = {
            **params,
            "ruled_tables_found": ruled,
            "text_rows": len(t.cells),
            "text_cols": t.n_cols(),
            "chars_text": len(md),
            "chars_flat": len(flat),
        }
    return info


def build_tables(levels: list[str], *, from_raw: bool) -> dict:
    stats: dict = {
        "clean": _copy_route(FOUR / "corpus/md", "clean"),
        "scan-clean": _copy_route(SIX / "corpus/glm2", "scan-clean"),
    }
    for level in levels:
        stats[f"scan-{level}"] = build_scan(level, from_raw=from_raw)
    stats["borderless"] = build_borderless()
    return stats


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", action="store_true")
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--from-raw", action="store_true", help="reuse cached glm-ocr replies")
    args = ap.parse_args(argv)
    if not (args.text or args.tables):
        args.text = args.tables = True
    setup_logging("rag_lab")
    CORPUS.mkdir(exist_ok=True)
    stats_path = CORPUS / "stats.json"
    stats = json.loads(stats_path.read_text("utf-8")) if stats_path.exists() else {}
    if args.text:
        stats["text"] = build_text()
        logger.info(
            "text track built",
            extra={
                "ctx": {
                    k: v.get("chars", v.get("chars_pypdf"))
                    for k, v in stats["text"].items()
                    if isinstance(v, dict)
                }
            },
        )
    if args.tables:
        stats["tables"] = build_tables(args.levels.split(","), from_raw=args.from_raw)
        logger.info("table track built", extra={"ctx": {"routes": list(stats["tables"])}})
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=1), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
