"""Three ways from a gazette PDF to text, one per route of the experiment (09's base and ocr,
a new rows route for table-only notices).

  base   01's pipeline untouched: pypdf page texts joined (the blind baseline)
  ocr    pages the health check flags (no text layer, or a (cid:N) garbage layer) replaced by
         glm-ocr's plain-text reading of the page image; every other page as base
  rows   every page where pdfplumber's lines strategy finds a table is replaced by one line per
         data row, each cell written 表頭：值 and cells joined with ｜ (03's key-value shape, the
         one that beat flattened text); other pages as ocr. Only table-type documents get it.

    uv run python experiments/13-points-tables/extract.py   -> corpus/text/<sysid>.<route>.txt
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

from rag_lab.log import setup_logging

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
CORPUS = HERE / "corpus"
DATA = ROOT / "data/13-points-tables"
sys.path.insert(0, str(HERE))
from doc_health import GARBAGE_PAGE, MIN_TEXT_CHARS, garbage_share, is_rotated  # noqa: E402

logger = logging.getLogger(__name__)

ROUTES = ("base", "ocr", "rows")
OCR_PROMPT = "Text Recognition:\n"  # glm-ocr reads the page as text (10 s/page on mini)
OCR_DPI = 300
_QUOTED = re.compile(r"「([^」]+)」")
_BARE = re.compile(
    r"(?:增訂|修正|訂定|廢止)([一-鿿]+?(?:自治條例|條例|法|規則|辦法|標準|準則|要點|規程|細則))"
)
_WS = re.compile(r"\s+")


def law_name(caption: str) -> str:
    """The regulation a gazette item is about: the first 「quoted」 name, else the name that
    follows 修正／增訂 in captions written without quotes (laws passed by the legislature)."""
    m = _QUOTED.search(caption) or _BARE.search(caption)
    return m.group(1) if m else ""


# -- base ---------------------------------------------------------------------------------------


def page_texts(pdf: Path | str) -> list[str]:
    return [p.extract_text() or "" for p in PdfReader(str(pdf)).pages]


def base_text(pdf: Path | str) -> str:
    return "\n".join(page_texts(pdf))


# -- ocr ----------------------------------------------------------------------------------------


def ocr_pages(pdf: Path | str) -> list[int]:
    """Page indices with no usable text layer: under MIN_TEXT_CHARS characters, or a
    (cid:N) share at or above GARBAGE_PAGE."""
    out = []
    with pdfplumber.open(pdf) as doc:
        for i, page in enumerate(doc.pages):
            text = page.extract_text() or ""
            if len(_WS.sub("", text)) < MIN_TEXT_CHARS or garbage_share(text) >= GARBAGE_PAGE:
                out.append(i)
    return out


def ocr_page(pdf: Path | str, page_no: int, cache: Path) -> str:
    """glm-ocr's text for one page, cached as <stem>-p<N>.txt under `cache`."""
    from ocr import glm_ocr

    pdf = Path(pdf)
    cache.mkdir(parents=True, exist_ok=True)
    txt = cache / f"{pdf.stem}-p{page_no}.txt"
    if txt.exists():
        return txt.read_text(encoding="utf-8")
    png = cache / f"{pdf.stem}-p{page_no}.png"
    if not png.exists():
        with pdfplumber.open(pdf) as doc:
            doc.pages[page_no].to_image(resolution=OCR_DPI, antialias=True).original.save(png)
    out = glm_ocr(png, prompt=OCR_PROMPT)
    txt.write_text(out, encoding="utf-8")
    logger.info("ocr page", extra={"ctx": {"pdf": pdf.name, "page": page_no, "chars": len(out)}})
    return out


def ocr_text(pdf: Path | str, cache: Path) -> str:
    pages = page_texts(pdf)
    for i in ocr_pages(pdf):
        pages[i] = ocr_page(pdf, i, cache)
    return "\n".join(pages)


# -- rows (tables as key-value lines) ---------------------------------------------------------------


def _clean(cell: object) -> str:
    return " ".join(str(cell or "").split())


def rows_lines(pdf: Path | str, page_no: int) -> list[str]:
    """Data rows of every ruled table on the page as 表頭：值｜表頭：值 lines; the first row of
    each table is its header, blank header cells inherit the cell to their left (merged header)."""
    out: list[str] = []
    with pdfplumber.open(pdf) as doc:
        page = doc.pages[page_no]
        if is_rotated(page):
            return []
        for table in page.extract_tables(
            {"vertical_strategy": "lines", "horizontal_strategy": "lines"}
        ):
            rows = [[_clean(c) for c in r] for r in table if r and any(_clean(c) for c in r)]
            if len(rows) < 2:
                continue
            header: list[str] = []
            for h in rows[0]:
                header.append(h or (header[-1] if header else ""))
            for r in rows[1:]:
                pairs = [f"{h}：{v}" for h, v in zip(header, r, strict=False) if v]
                if pairs:
                    out.append("｜".join(pairs))
    return out


def rows_text(pdf: Path | str, pages: list[str] | None = None) -> str:
    """Table pages replaced by their rows lines; every other page as given (default pypdf)."""
    pages = page_texts(pdf) if pages is None else list(pages)
    out = []
    for i, text in enumerate(pages):
        lines = rows_lines(pdf, i)
        out.append("\n".join(lines) if lines else text)
    return "\n".join(out)


# -- all routes for the corpus -------------------------------------------------------------------


def main(argv: list[str]) -> int:
    """--no-ocr: write only the base route for documents that have pages to OCR (so the text
    can be built while ollama is busy); run again without it to add ocr and rows."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ocr", action="store_true")
    args = ap.parse_args(argv)
    setup_logging("rag_lab")
    docs = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))["docs"]
    (CORPUS / "text").mkdir(exist_ok=True)
    stats = {}
    for d in docs:
        pdf = CORPUS / "pdf" / f"{d['sysid']}.pdf"
        pages = page_texts(pdf)
        ocr = ocr_pages(pdf)
        if ocr and args.no_ocr:
            (CORPUS / "text" / f"{d['sysid']}.base.txt").write_text(
                base_text(pdf), encoding="utf-8"
            )
            logger.info(
                "base only (ocr deferred)", extra={"ctx": {"sysid": d["sysid"], "pages": ocr}}
            )
            continue
        for i in ocr:
            pages[i] = ocr_page(pdf, i, DATA / "ocr")
        texts = {"base": base_text(pdf), "ocr": "\n".join(pages)}
        texts["rows"] = rows_text(pdf, pages=pages) if d["type"] == "tables" else texts["ocr"]
        for route, text in texts.items():
            (CORPUS / "text" / f"{d['sysid']}.{route}.txt").write_text(text, encoding="utf-8")
        n_rows = (
            sum(len(rows_lines(pdf, i)) for i in range(len(pages))) if d["type"] == "tables" else 0
        )
        stats[d["sysid"]] = {
            "ocr_pages": ocr,
            "rows": n_rows,
            "chars": {k: len(v) for k, v in texts.items()},
        }
    old = CORPUS / "extract_stats.json"
    merged = json.loads(old.read_text("utf-8")) if old.exists() else {}
    merged.update(stats)
    old.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("extracted", extra={"ctx": {"docs": len(docs)}})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
