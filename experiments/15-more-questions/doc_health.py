"""Data health check (08's, plus garbage detection): point it at a folder of PDFs before anyone
has told you how messy they are.

    uv run python experiments/09-messy-real/doc_health.py <folder> [-o health.csv]

One row per PDF (pdfplumber only, no model):
  pages               page count
  text_pages          pages with a text layer (>= 20 characters); the rest need OCR
  table_pages         pages where the ruled-lines strategy finds a table
  aligned_text_pages  pages with no ruled table but >= 5 rows of wide-gapped words sharing
                      >= 3 x positions: a table typeset without lines (Word exports do this)
  two_column_pages    pages whose words sit in two column-wide clusters left/right of the
                      middle and almost none cross it: pdfplumber will interleave the columns
  repeated_lines      lines (digits folded to #) in the top/bottom 15% of the page, outside any
                      ruled table, that recur at the same height on >= 50% of pages (3 pages or
                      more): headers, footers, page numbers that end up in chunks
  garbage_pages       pages with a text layer that is mostly (cid:N) tokens: an embedded font
                      with no ToUnicode map (09 met one in a gazette); such pages need OCR
  garbage_share       (cid:N) tokens / (tokens + real characters) over the whole file
  rotated_pages       pages stored with a /Rotate or drawn sideways (under half the characters
                      upright): a landscape 附表; pdfplumber reads them backwards, pypdf and
                      PyMuPDF do not
The real gazettes 09 sampled are the test cases (tests/test_doc_health.py).
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import pdfplumber

logger = logging.getLogger(__name__)

MIN_TEXT_CHARS = 20
EDGE = 0.15  # top/bottom band for header/footer candidates
REPEAT_SHARE = 0.5
GAP_EM = 1.5  # a column gap, in em
COLUMN_SHARE = 0.3  # a text column spans at least this share of the page width
SAME_TOP = 6.0  # pt: a header sits at the same height on every page
MIN_PAGES_FOR_REPEAT = 3
GARBAGE_PAGE = 0.2  # a page whose text is >= this share of (cid:N) tokens is unreadable
_DIGITS = re.compile(r"\d+")
_CID = re.compile(r"\(cid:\d+\)")


def garbage_share(text: str) -> float:
    """Share of (cid:N) tokens among tokens + real (non-space) characters; 0 for empty text."""
    n_cid = len(_CID.findall(text))
    real = sum(1 for ch in _CID.sub("", text) if not ch.isspace())
    return n_cid / (n_cid + real) if n_cid + real else 0.0


def _fold(line: str) -> str:
    return _DIGITS.sub("#", line.strip())


def _is_wordy(key: str) -> bool:
    """At least two real characters: '# # #' is a numeric table row, not a header."""
    return sum(ch not in "# .,-" for ch in key) >= 2


def is_rotated(page: pdfplumber.page.Page) -> bool:
    chars = page.chars
    upright = sum(1 for c in chars if c.get("upright", True))
    return page.rotation % 360 != 0 or (bool(chars) and upright < 0.5 * len(chars))


def _aligned_rows(words: list[dict]) -> bool:
    """>= 5 rows that each have >= 2 wide gaps (>= GAP_EM em) between words, and >= 3 word
    x positions shared by >= 5 of those rows. Prose lines like 「第 1 條」 have 3 words but
    only one-space gaps, so they do not count."""
    rows: dict[int, list[dict]] = {}
    for w in words:
        rows.setdefault(round(w["top"] / 3), []).append(w)
    tabular = []
    for r in rows.values():
        r = sorted(r, key=lambda w: w["x0"])
        em = max(w["bottom"] - w["top"] for w in r)
        gaps = sum(1 for a, b in zip(r, r[1:], strict=False) if b["x0"] - a["x1"] >= GAP_EM * em)
        if gaps >= 2:
            tabular.append(r)
    if len(tabular) < 5:
        return False
    x_counts = Counter(round(w["x0"] / 4) for r in tabular for w in r)
    return sum(1 for n in x_counts.values() if n >= 5) >= 3


def _two_column(words: list[dict], width: float) -> bool:
    """Words split into a left and a right group, almost none crossing the middle, and each
    group spans a text column's width (a table's narrow number column does not)."""
    if len(words) < 20:
        return False
    mid = width / 2
    left = [w for w in words if w["x1"] <= mid]
    right = [w for w in words if w["x0"] >= mid]
    crossing = len(words) - len(left) - len(right)
    if not (
        len(left) >= 0.3 * len(words)
        and len(right) >= 0.3 * len(words)
        and crossing <= 0.1 * len(words)
    ):
        return False
    span = lambda ws: max(w["x1"] for w in ws) - min(w["x0"] for w in ws)  # noqa: E731
    return span(left) >= COLUMN_SHARE * width and span(right) >= COLUMN_SHARE * width


def _same_height(tops: list[float]) -> int:
    """How many occurrences sit within SAME_TOP of the median: a header is always at the same
    height, a 「第 # 條」 line that happens to fall in the band is not."""
    med = sorted(tops)[len(tops) // 2]
    return sum(1 for t in tops if abs(t - med) <= SAME_TOP)


def inspect_pdf(path: Path | str) -> dict:
    path = Path(path)
    out = {
        "file": path.name,
        "pages": 0,
        "text_pages": 0,
        "table_pages": 0,
        "aligned_text_pages": 0,
        "two_column_pages": 0,
        "repeated_lines": 0,
        "repeated_examples": "",
        "garbage_pages": 0,
        "garbage_share": 0.0,
        "rotated_pages": 0,
    }
    edge_lines: dict[str, list[float]] = {}
    n_cid = n_real = 0
    with pdfplumber.open(path) as pdf:
        out["pages"] = len(pdf.pages)
        for page in pdf.pages:
            text = page.extract_text() or ""
            if len(text.replace(" ", "").replace("\n", "")) < MIN_TEXT_CHARS:
                continue
            out["text_pages"] += 1
            out["rotated_pages"] += is_rotated(page)
            share = garbage_share(text)
            n_cid += len(_CID.findall(text))
            n_real += sum(1 for ch in _CID.sub("", text) if not ch.isspace())
            if share >= GARBAGE_PAGE:
                out["garbage_pages"] += 1
                continue  # the rest of the checks read text, and this page has none to read
            words = page.extract_words()
            has_table = bool(page.find_tables())
            out["table_pages"] += has_table
            aligned = not has_table and _aligned_rows(words)
            out["aligned_text_pages"] += aligned
            if not has_table and not aligned and _two_column(words, page.width):
                out["two_column_pages"] += (
                    1  # a wide lineless table is reported as aligned, not as two columns
                )
            h = page.height
            boxes = [t.bbox for t in page.find_tables()] if has_table else []
            seen: set[str] = set()
            for ln in page.extract_text_lines():
                if ln["top"] > EDGE * h and ln["bottom"] < (1 - EDGE) * h:
                    continue
                if any(b[1] <= ln["top"] and ln["bottom"] <= b[3] for b in boxes):
                    continue  # a table row, not page furniture
                key = _fold(ln["text"])
                if _is_wordy(key) and key not in seen:
                    seen.add(key)
                    edge_lines.setdefault(key, []).append(ln["top"])
    out["garbage_share"] = round(n_cid / (n_cid + n_real), 3) if n_cid + n_real else 0.0
    n = out["text_pages"] - out["garbage_pages"]
    if n >= MIN_PAGES_FOR_REPEAT:
        rep = [k for k, tops in edge_lines.items() if _same_height(tops) >= REPEAT_SHARE * n]
        out["repeated_lines"] = len(rep)
        out["repeated_examples"] = " | ".join(sorted(rep)[:5])
    logger.info(
        "inspected", extra={"ctx": {k: v for k, v in out.items() if k != "repeated_examples"}}
    )
    return out


def main(argv: list[str]) -> int:
    from rag_lab.log import setup_logging

    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("-o", "--out", default="health.csv")
    args = ap.parse_args(argv)
    setup_logging("rag_lab")
    rows = [inspect_pdf(p) for p in sorted(Path(args.folder).rglob("*.pdf"))]
    if not rows:
        logger.warning("no PDFs", extra={"ctx": {"folder": args.folder}})
        return 1
    with Path(args.out).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    logger.info("health written", extra={"ctx": {"files": len(rows), "out": args.out}})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
