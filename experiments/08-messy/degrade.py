"""Make the clean regulation messy, one kind of mess at a time. Every function returns the
parameters it used so run.py can put them in meta.json.

  header_footer   stamp a title / date / "第 N 頁／共 M 頁" on every page of an existing PDF
  typeset         re-typeset lines into an A4 n-column layout (two_column = what pdfplumber interleaves)
  concat          append a second PDF (a same-domain distractor)
  degrade_png     print-and-scan simulation on a page image, two levels (phone / copy3)
  borderless_table  typeset one of 04's tables with no ruling lines at all

PyMuPDF's built-in "cjk" font (Droid Sans Fallback) covers Traditional Chinese, so nothing has
to be installed on the machine.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

A4 = pymupdf.paper_rect("a4")  # 595 x 842 pt
MARGIN = 50.0
GUTTER = 20.0
FONT_SIZE = 10  # pymupdf wants an int
LEADING = 1.5
MIN_LINE = 8  # a typeset line shorter than this is not used as a label witness
_BW = [0] * 128 + [255] * 128  # Image.point lookup: re-binarise after resampling / JPEG

LEVELS: dict[str, dict] = {
    "phone": {
        "rotate_deg": 1.5,
        "blur_sigma": 1.0,
        "shadow": 0.25,
        "jpeg_quality": 60,
        "dpi_scale": 200 / 300,
        "salt_pepper": 0.0,
        "binarize": None,
    },
    "copy3": {
        "rotate_deg": 1.5,
        "blur_sigma": 1.0,
        "shadow": 0.25,
        "jpeg_quality": 60,
        "dpi_scale": 150 / 300,
        "salt_pepper": 0.01,
        "binarize": 160,
    },
}


def _font() -> pymupdf.Font:
    return pymupdf.Font("cjk")


def _write(page: pymupdf.Page, pos: tuple[float, float], text: str, *, size: int) -> None:
    tw = pymupdf.TextWriter(page.rect)
    tw.append(pymupdf.Point(*pos), text, font=_font(), fontsize=size)
    tw.write_text(page)


# -- header / footer ---------------------------------------------------------------------------


def header_footer(
    src: Path | str,
    dst: Path | str,
    *,
    title: str,
    date: str = "修正日期：民國 114 年 5 月 9 日",
    size: int = 9,
) -> dict:
    doc = pymupdf.open(src)
    n = len(doc)
    font = _font()
    for i in range(1, n + 1):
        page = doc[i - 1]
        w = page.rect.width
        h = page.rect.height
        _write(page, (MARGIN, 30), title, size=size)
        _write(page, (w - MARGIN - font.text_length(date, fontsize=size), 30), date, size=size)
        foot = f"第 {i} 頁／共 {n} 頁"
        _write(page, (w - MARGIN - font.text_length(foot, fontsize=size), h - 20), foot, size=size)
    doc.save(dst)
    doc.close()
    logger.info("header_footer", extra={"ctx": {"src": Path(src).name, "pages": n}})
    return {"pages": n, "title": title, "date": date, "size": size}


# -- two columns -------------------------------------------------------------------------------


def _wrap(text: str, width: float, font: pymupdf.Font, size: int) -> list[str]:
    """Greedy character wrap (CJK has no word boundaries) to `width` points."""
    out: list[str] = []
    buf = ""
    for ch in text:
        if buf and font.text_length(buf + ch, fontsize=size) > width:
            out.append(buf)
            buf = ch
        else:
            buf += ch
    out.append(buf)
    return out


def two_column(items: list[tuple[str, str]], dst: Path | str) -> list[tuple[str, str]]:
    return typeset(items, dst, columns=2)


def typeset(
    items: list[tuple[str, str]],
    dst: Path | str,
    *,
    columns: int,
    size: int = FONT_SIZE,
) -> list[tuple[str, str]]:
    """items = (line, label). Lines are wrapped to the column width and laid out column by
    column, page by page. Returns the typeset lines with their labels (the line map labels.py
    uses); the PDF content stream is in that same order."""
    font = _font()
    col_w = (A4.width - 2 * MARGIN - GUTTER * (columns - 1)) / columns
    lead = size * LEADING
    per_col = int((A4.height - 2 * MARGIN) // lead)
    out: list[tuple[str, str]] = []
    for line, label in items:
        for piece in _wrap(line, col_w, font, size) if line.strip() else [""]:
            out.append((piece, label))
    doc = pymupdf.open()
    per_page = columns * per_col
    for start in range(0, len(out), per_page):
        page = doc.new_page(width=A4.width, height=A4.height)
        tw = pymupdf.TextWriter(page.rect)
        for k, (piece, _) in enumerate(out[start : start + per_page]):
            col, row = divmod(k, per_col)
            x = MARGIN + col * (col_w + GUTTER)
            y = MARGIN + (row + 1) * lead
            if piece:
                tw.append(pymupdf.Point(x, y), piece, font=font, fontsize=size)
        tw.write_text(page)
    doc.save(dst)
    n_pages = len(doc)
    doc.close()
    logger.info(
        "typeset",
        extra={
            "ctx": {
                "columns": columns,
                "lines_in": len(items),
                "lines_out": len(out),
                "pages": n_pages,
            }
        },
    )
    return out


# -- concat ------------------------------------------------------------------------------------


def concat(a: Path | str, b: Path | str, dst: Path | str) -> dict:
    doc = pymupdf.open(a)
    n_a = len(doc)
    other = pymupdf.open(b)
    doc.insert_pdf(other)
    n = len(doc)
    doc.save(dst)
    doc.close()
    other.close()
    return {"pages": n, "pages_a": n_a, "pages_b": n - n_a}


# -- scan simulation ---------------------------------------------------------------------------


def degrade_png(src: Path | str, dst: Path | str, *, level: str, seed: int = 0) -> dict:
    """Pillow-only print-and-scan: rotate, blur, uneven lighting, JPEG, lower dpi; copy3 adds
    salt-and-pepper and binarisation (a third-generation photocopy). Operations follow the list
    OHRBench used for its semantic-noise levels."""
    p = LEVELS[level]
    rng = np.random.default_rng(seed)
    im = Image.open(src).convert("L")
    w, h = im.size
    im = im.rotate(p["rotate_deg"], resample=Image.Resampling.BICUBIC, fillcolor=255)
    if p["blur_sigma"]:
        im = im.filter(ImageFilter.GaussianBlur(p["blur_sigma"]))
    arr = np.asarray(im, dtype=np.float32)
    if p["shadow"]:
        ramp = np.linspace(1.0, 1.0 - p["shadow"], w, dtype=np.float32)
        arr = arr * ramp[None, :]
    if p["salt_pepper"]:
        mask = rng.random(arr.shape) < p["salt_pepper"]
        arr[mask] = rng.choice([0.0, 255.0], size=int(mask.sum()))
    if p["binarize"] is not None:
        arr = np.where(arr < p["binarize"], 0.0, 255.0)
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    new_size = (round(w * p["dpi_scale"]), round(h * p["dpi_scale"]))
    im = im.resize(new_size, resample=Image.Resampling.LANCZOS)
    if p["binarize"] is not None:
        im = im.point(_BW)
    if p["jpeg_quality"]:
        tmp = Path(dst).with_suffix(".jpg")
        im.save(tmp, quality=p["jpeg_quality"])
        im = Image.open(tmp).convert("L")
        if p["binarize"] is not None:
            im = im.point(_BW)
        im.save(dst)
        tmp.unlink()
    else:
        im.save(dst)
    return {"level": level, "seed": seed, **p}


# -- borderless table --------------------------------------------------------------------------


def borderless_table(t: dict, dst: Path | str, *, size: int = 9, min_size: int = 6) -> dict:
    """04's table JSON (title, names, rows, notes) typeset as aligned text columns with no lines.
    Column width = the widest cell; the font shrinks until the table fits the page width."""
    font = _font()
    cols = list(zip(t["names"], *t["rows"], strict=True))
    gap = 12.0

    def widths(s: int) -> list[float]:
        return [max(font.text_length(str(c), fontsize=s) for c in col) + gap for col in cols]

    while size > min_size and sum(widths(size)) > A4.width - 2 * MARGIN:
        size -= 1
    ws = widths(size)
    lead = size * LEADING
    doc = pymupdf.open()
    page = doc.new_page(width=A4.width, height=A4.height)
    tw = pymupdf.TextWriter(page.rect)
    y = MARGIN
    tw.append(pymupdf.Point(MARGIN, y), t["title"], font=font, fontsize=size + 2)
    y += lead * 2
    for row in [t["names"], *t["rows"]]:
        if y > A4.height - MARGIN:
            tw.write_text(page)
            page = doc.new_page(width=A4.width, height=A4.height)
            tw = pymupdf.TextWriter(page.rect)
            y = MARGIN
        x = MARGIN
        for cell, w in zip(row, ws, strict=True):
            if cell:
                tw.append(pymupdf.Point(x, y), str(cell), font=font, fontsize=size)
            x += w
        y += lead
    if t.get("notes"):
        y += lead
        for ln in _wrap(t["notes"], A4.width - 2 * MARGIN, font, size):
            if y > A4.height - MARGIN:
                tw.write_text(page)
                page = doc.new_page(width=A4.width, height=A4.height)
                tw = pymupdf.TextWriter(page.rect)
                y = MARGIN
            tw.append(pymupdf.Point(MARGIN, y), ln, font=font, fontsize=size)
            y += lead
    tw.write_text(page)
    doc.save(dst)
    n = len(doc)
    doc.close()
    return {"font_size": size, "pages": n, "cols": len(cols)}
