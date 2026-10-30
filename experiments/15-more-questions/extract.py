"""The fourth extraction route for experiment 10: 09's `ocr` and `rows` fixes stacked.

09's three routes (base / ocr / rows) are read from ../09-messy-real/corpus/text; this file is a
copy of 09's extract.py whose main only writes corpus/text/<sysid>.ocrrows.txt.

    uv run python experiments/10-fix-ablation/extract.py
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
CORPUS09 = HERE.parent / "09-messy-real" / "corpus"
DATA = ROOT / "data/09-messy-real"
sys.path.insert(0, str(HERE))
from chunkers import _HEAD_CN, _MARK_CN, cn_to_int, int_to_cn  # noqa: E402
from doc_health import GARBAGE_PAGE, MIN_TEXT_CHARS, garbage_share, is_rotated  # noqa: E402

logger = logging.getLogger(__name__)

ROUTES = ("base", "ocr", "rows")
OCR_PROMPT = "Text Recognition:\n"  # glm-ocr reads the page as text (10 s/page on mini)
OCR_DPI = 300
WIDTH_TOL = 0.15  # a table this much narrower/wider than the first comparison table is not it
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


# -- rows ---------------------------------------------------------------------------------------


def _is_header_cell(s: str) -> bool:
    s = _WS.sub("", s)
    return 0 < len(s) <= 8 and (s.startswith(("修正", "現行", "新增")) or s == "說明")


def _cell_text(page: pdfplumber.page.Page, bbox: tuple) -> str:
    text = page.crop(bbox).extract_text() or ""
    return "\n".join(ln.strip() for ln in text.splitlines() if ln.strip())


def _split_header(s: str) -> tuple[str, str] | None:
    """('14-1', rest) when the cell starts with an article header, ('ch:…', '') for a chapter
    or section line, else None."""
    m = _HEAD_CN.match(s)
    if m:
        num = cn_to_int(m.group(1))
        label = f"{num}-{cn_to_int(m.group(2))}" if m.group(2) else str(num)
        return label, s[m.end() :].strip()
    if _MARK_CN.match(s):
        return "ch:" + s, ""
    return None


def _column_edges(page: pdfplumber.page.Page, t: pdfplumber.table.Table) -> list[float] | None:
    """The three column left edges of a comparison table: the table's three cell positions, or
    the positions of its three header cells when a column is drawn as sub-columns."""
    lefts = sorted({round(b[0]) for row in t.rows for b in row.cells if b})
    if len(lefts) == 3:
        return [float(x) for x in lefts]
    for row in t.rows:
        heads = [b[0] for b in row.cells if b and _is_header_cell(_cell_text(page, b))]
        if len(heads) == 3:
            return sorted(float(x) for x in heads)
    return None


def compare_rows(pdf: Path | str) -> list[dict]:
    """One record per article of the draft's comparison table:
    {label, amended, current, note, pages}. The first table with three column positions sets
    the column edges; later tables of about the same width are read against them, each cell
    going to the nearest column (a 說明 drawn as two sub-columns folds back). Header rows are
    dropped; a row whose 修正 or 現行 cell does not start with 第…條 continues the article
    before it (a cell cut by the page break)."""
    records: list[dict] = []
    ref: list[float] | None = None
    ref_width = 0.0
    with pdfplumber.open(pdf) as doc:
        for page in doc.pages:
            if is_rotated(page):
                continue  # pdfplumber reads these backwards; the base route keeps pypdf's text
            for t in page.find_tables():
                width = t.bbox[2] - t.bbox[0]
                if ref is None:
                    ref = _column_edges(page, t)
                    if ref is None:
                        continue
                    ref_width = width
                elif abs(width - ref_width) > WIDTH_TOL * ref_width:
                    continue
                for row in t.rows:
                    cols: list[list[str]] = [[], [], []]
                    for b in row.cells:
                        if b is None:
                            continue
                        text = _cell_text(page, b)
                        if text:
                            k = min(range(3), key=lambda k: abs(b[0] - ref[k]))  # type: ignore[index]
                            cols[k].append(text)
                    cells = ["\n".join(c) for c in cols]
                    if not any(cells):
                        continue
                    if all(_is_header_cell(c) for c in cells if c):
                        continue
                    head = _split_header(cells[0]) or _split_header(cells[1])
                    if head:
                        label, _ = head
                        amended = _split_header(cells[0])
                        current = _split_header(cells[1])
                        records.append(
                            {
                                "label": label,
                                "amended": amended[1] if amended else cells[0],
                                "current": current[1] if current else cells[1],
                                "note": cells[2],
                                "pages": [page.page_number - 1],
                            }
                        )
                    elif records:
                        r = records[-1]
                        for k, f in enumerate(("amended", "current", "note")):
                            if cells[k]:
                                r[f] = f"{r[f]}\n{cells[k]}" if r[f] else cells[k]
                        if r["pages"][-1] != page.page_number - 1:
                            r["pages"].append(page.page_number - 1)
    logger.info("compare rows", extra={"ctx": {"pdf": Path(pdf).name, "records": len(records)}})
    return records


def _cn_label(label: str) -> str:
    num, _, sub = label.partition("-")
    return f"第{int_to_cn(int(num))}條" + (f"之{int_to_cn(int(sub))}" if sub else "")


def records_to_text(records: list[dict]) -> str:
    """Records as text the Chinese-numeral parser chunks one article per block."""
    out = []
    for r in records:
        if r["label"].startswith("ch:"):
            out.append(r["label"][3:])
            continue
        out.append(
            f"{_cn_label(r['label'])} 修正條文：\n{r['amended']}\n"
            f"現行條文：\n{r['current']}\n"
            f"說明：\n{r['note']}"
        )
    return "\n".join(out) + "\n"


def rows_text(pdf: Path | str, pages: list[str] | None = None) -> str:
    """Pages holding the comparison table replaced (once, where it starts) by the records;
    every other page as given (`pages`, default the pypdf text)."""
    records = compare_rows(pdf)
    table_pages = {p for r in records for p in r["pages"]}
    pages = page_texts(pdf) if pages is None else list(pages)
    out = []
    inserted = False
    for i, text in enumerate(pages):
        if i in table_pages:
            if not inserted:
                out.append(records_to_text(records))
                inserted = True
        else:
            out.append(text)
    return "\n".join(out)


def ocrrows_text(pdf: Path | str, cache: Path) -> str:
    """09's two fixes stacked: OCR pages swapped in first, then the comparison table rebuilt.
    In 09's corpus the scanned/garbage docs have no table and the table docs no OCR page, so
    this equals `ocr` on the former and `rows` on the latter (tests/test_extract.py)."""
    pages = page_texts(pdf)
    for i in ocr_pages(pdf):
        pages[i] = ocr_page(pdf, i, cache)
    return rows_text(pdf, pages=pages)


# -- the one new route for the corpus (the other three are read from 09) --------------------------


def main() -> int:
    setup_logging("rag_lab")
    docs = json.loads((CORPUS09 / "manifest.json").read_text(encoding="utf-8"))["docs"]
    (CORPUS / "text").mkdir(parents=True, exist_ok=True)
    for d in docs:
        pdf = CORPUS09 / "pdf" / f"{d['sysid']}.pdf"
        text = (
            ocrrows_text(pdf, DATA / "ocr") if d["type"] == "draft" else ocr_text(pdf, DATA / "ocr")
        )
        (CORPUS / "text" / f"{d['sysid']}.ocrrows.txt").write_text(text, encoding="utf-8")
    logger.info("extracted", extra={"ctx": {"docs": len(docs), "route": "ocrrows"}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
