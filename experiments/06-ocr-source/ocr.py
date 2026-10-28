"""Three ways from a page image to a markdown table.

  vision     macOS Vision (ocrmac): line segments with boxes -> rows by y, cells by x
  tesseract  tesseract chi_tra+eng, --psm 6, TSV words -> segments by gap -> same clustering
  glm        glm-ocr via the host ollama: "Table Recognition:" -> markdown (or HTML, or plain
             text when it finds no table) -> markdown
  glm2       the same model and prompt with one trailing newline -> HTML with spans -> markdown

The first two recover no column structure beyond "what sits on the same line"; glm-ocr returns
an HTML table with spans. All three keep the title line and the 註 lines.
"""

from __future__ import annotations

import base64
import csv
import io
import logging
import re
import statistics
import subprocess
import unicodedata
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

from PIL import Image

from rag_lab import ollama

logger = logging.getLogger(__name__)

_TITLE = re.compile(r"^表[^\s]+")
_NOTE = re.compile(r"^\s*註")
GLM_MODEL = "glm-ocr"
GLM_PROMPT = "Table Recognition:"  # as the ollama page shows it
GLM_PROMPT_NL = "Table Recognition:\n"  # one trailing newline: the model then answers in HTML


@dataclass
class Seg:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float = 1.0

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def h(self) -> float:
        return self.y1 - self.y0


# -- segment sources ----------------------------------------------------------------------------


def vision_segments(png: Path | str) -> list[Seg]:
    from ocrmac import ocrmac

    w, h = Image.open(png).size
    out = []
    res = ocrmac.OCR(
        str(png), language_preference=["zh-Hant", "en-US"], recognition_level="accurate"
    ).recognize()
    for text, conf, (x, y, bw, bh) in res:  # normalised, origin bottom-left
        out.append(Seg(text.strip(), x * w, (1 - y - bh) * h, (x + bw) * w, (1 - y) * h, conf))
    return out


def tesseract_segments(png: Path | str, *, gap_factor: float = 1.0) -> list[Seg]:
    """Words from `tesseract ... tsv`, joined into a segment while the gap to the next word is
    under `gap_factor` x the line height."""
    tsv = subprocess.run(
        ["tesseract", str(png), "-", "-l", "chi_tra+eng", "--psm", "6", "tsv"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    words = [
        r
        for r in csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE)
        if r["text"].strip() and float(r["conf"]) >= 0
    ]
    lines: dict[tuple[str, str, str], list[dict]] = {}
    for r in words:
        lines.setdefault((r["block_num"], r["par_num"], r["line_num"]), []).append(r)
    out = []
    for ws in lines.values():
        ws.sort(key=lambda r: int(r["left"]))
        cur: list[dict] = []
        for r in ws:
            if cur:
                prev = cur[-1]
                gap = int(r["left"]) - (int(prev["left"]) + int(prev["width"]))
                if gap > gap_factor * int(prev["height"]):
                    out.append(_tess_seg(cur))
                    cur = []
            cur.append(r)
        if cur:
            out.append(_tess_seg(cur))
    return out


def _tess_seg(ws: list[dict]) -> Seg:
    x0 = min(int(r["left"]) for r in ws)
    x1 = max(int(r["left"]) + int(r["width"]) for r in ws)
    y0 = min(int(r["top"]) for r in ws)
    y1 = max(int(r["top"]) + int(r["height"]) for r in ws)
    # tesseract chi_tra puts a space between every CJK character; glue them back
    text = re.sub(r"(?<=[　-鿿＀-￯])\s+(?=[　-鿿＀-￯])", "", " ".join(r["text"] for r in ws))
    return Seg(text, x0, y0, x1, y1, min(float(r["conf"]) for r in ws) / 100)


# -- segments -> table text ---------------------------------------------------------------------


def cluster_rows(segs: list[Seg], *, tol_factor: float = 0.6) -> list[list[Seg]]:
    """Group segments whose vertical centres are within tol of the row's; cells left-to-right."""
    if not segs:
        return []
    tol = tol_factor * statistics.median(s.h for s in segs)
    rows: list[list[Seg]] = []
    for s in sorted(segs, key=lambda s: s.yc):
        if rows and abs(s.yc - statistics.mean(r.yc for r in rows[-1])) <= tol:
            rows[-1].append(s)
        else:
            rows.append([s])
    for r in rows:
        r.sort(key=lambda s: s.x0)
    return rows


def rows_to_markdown(rows: list[list[Seg]]) -> str:
    """Title lines first, then one markdown row per clustered row, 註 lines last as prose."""
    title, body, notes = [], [], []
    for r in rows:
        cells = [s.text for s in r]
        if not body and _TITLE.match(cells[0]):
            title.append("".join(cells))
        elif _NOTE.match(cells[0]) or (notes and len(cells) == 1 and not body[-1:] == []):
            notes.append(" ".join(cells))
        elif notes:  # a line after the notes started: still notes (wrapped)
            notes.append(" ".join(cells))
        else:
            body.append("| " + " | ".join(cells) + " |")
    out = [" ".join(title)] if title else []
    if body:
        n = body[0].count("|") - 1
        out.append(body[0])
        out.append("|" + "---|" * n)
        out.extend(body[1:])
    if notes:
        out.append("".join(notes))
    return "\n".join(out)


def segments_to_markdown(segs: list[Seg]) -> str:
    return rows_to_markdown(cluster_rows(segs))


# -- glm-ocr ------------------------------------------------------------------------------------


def glm_ocr(
    png: Path | str, *, prompt: str = GLM_PROMPT, model: str = GLM_MODEL, timeout: int = 600
) -> str:
    """Raw reply of glm-ocr's table-recognition prompt (HTML, markdown or plain text)."""
    b64 = base64.b64encode(Path(png).read_bytes()).decode()
    r = ollama.chat(
        prompt,
        model=model,
        images=[b64],
        options={"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 4096},
        timeout=timeout,
    )
    return r["content"]


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[tuple[str, int, int]]] = []
        self.cur: list[tuple[str, int, int]] | None = None
        self.cell: list[str] | None = None
        self.span = (1, 1)
        self.other: list[str] = []
        self._text: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr":
            self.cur = []
        elif tag in ("td", "th"):
            self.cell = []
            self.span = (int(a.get("rowspan", 1) or 1), int(a.get("colspan", 1) or 1))
        elif tag == "br" and self.cell is not None:
            self.cell.append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None and self.cur is not None:
            self.cur.append(("".join(self.cell).strip(), *self.span))
            self.cell = None
        elif tag == "tr" and self.cur is not None:
            self.rows.append(self.cur)
            self.cur = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)
        elif self.cur is None:
            self.other.append(data)


def html_to_markdown(html: str) -> str:
    """Expand rowspan/colspan into a full grid (spanned cells repeat the text) and render."""
    p = _TableParser()
    p.feed(html)
    grid: list[list[str | None]] = []
    for i, row in enumerate(p.rows):
        while len(grid) <= i:
            grid.append([])
        j = 0
        for text, rs, cs in row:
            while j < len(grid[i]) and grid[i][j] is not None:
                j += 1
            for di in range(rs):
                while len(grid) <= i + di:
                    grid.append([])
                line = grid[i + di]
                while len(line) < j + cs:
                    line.append(None)
                for dj in range(cs):
                    line[j + dj] = text
            j += cs
    width = max((len(r) for r in grid), default=0)
    rows = [[(c or "") for c in r] + [""] * (width - len(r)) for r in grid if any(r)]
    other = " ".join(t.strip() for t in p.other if t.strip())
    title = [ln for ln in other.splitlines() if _TITLE.match(ln.strip())]
    body, notes = [], []
    for r in rows:
        if _NOTE.match(r[0]) and len(set(r)) <= 2:
            notes.append(r[0])
        else:
            body.append("| " + " | ".join(r) + " |")
    out = [title[0].strip()] if title else []
    if body:
        out.append(body[0])
        out.append("|" + "---|" * width)
        out.extend(body[1:])
    if notes:
        out.append("".join(notes))
    if other and not title:
        out.append(other)
    return "\n".join(out)


def glm_to_markdown(raw: str) -> str:
    """glm-ocr through ollama answers in markdown; HTML gets expanded, anything else is kept
    as it came (plain text means the model saw no table -- that is the result, not a bug)."""
    if "<table" in raw:
        return html_to_markdown(raw)
    return raw.strip()


def norm(s: str) -> str:
    """For comparing cell text across sources: NFKC, no spaces, ASCII punctuation."""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", "", s)
    return s.replace("，", ",").replace("。", ".").replace("（", "(").replace("）", ")")
