"""glm-ocr's HTML table -> markdown, copied from 23 (06's renderer) with one switch added:
flatten=True joins the rows of a multi-level <thead> (one with a cell spanning columns) into one
header row the way 04 named the columns of the text layer (top to bottom, a part equal to the one
above it is skipped, joined with ／). With flatten=False the output is 23's, byte for byte.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from html.parser import HTMLParser

logger = logging.getLogger(__name__)

_TITLE = re.compile(r"^表[^\s]+")
_NOTE = re.compile(r"^\s*註")


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[tuple[str, int, int]]] = []
        self.cur: list[tuple[str, int, int]] | None = None
        self.cell: list[str] | None = None
        self.span = (1, 1)
        self.other: list[str] = []
        self._text: list[str] = []
        self.in_head = False
        self.head = 0  # rows inside <thead>
        self.head_spans = False  # a <thead> cell spans columns: the rows are a header hierarchy

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr":
            self.cur = []
        elif tag in ("td", "th"):
            self.cell = []
            self.span = (int(a.get("rowspan", 1) or 1), int(a.get("colspan", 1) or 1))
            self.head_spans |= self.in_head and self.span[1] > 1
        elif tag == "br" and self.cell is not None:
            self.cell.append(" ")
        elif tag == "thead":
            self.in_head = True

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None and self.cur is not None:
            self.cur.append(("".join(self.cell).strip(), self.span[0], self.span[1]))
            self.cell = None
        elif tag == "tr" and self.cur is not None:
            self.rows.append(self.cur)
            self.cur = None
            self.head += self.in_head
        elif tag == "thead":
            self.in_head = False

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)
        elif self.cur is None:
            self.other.append(data)


def header_rows(html: str) -> int:
    """How many rows glm-ocr put in <thead> (0 for plain text or a table without one)."""
    p = _TableParser()
    p.feed(html)
    return p.head


def _flat_header(rows: list[list[str | None]]) -> list[str | None]:
    """04's column names: per column, the header cells top to bottom, a repeat skipped, ／."""
    names: list[str | None] = []
    for j in range(max(len(r) for r in rows)):
        parts: list[str] = []
        for r in rows:
            t = r[j] if j < len(r) else None
            if t and (not parts or parts[-1] != t):
                parts.append(t)
        names.append("／".join(parts))
    return names


def html_to_markdown(html: str, *, flatten: bool = False) -> str:
    """Expand rowspan/colspan into a full grid (spanned cells repeat the text) and render.
    flatten: a <thead> of more than one row with a cell spanning columns becomes one row of
    04-style column names. Without such a cell the rows are no hierarchy: glm-ocr put the first
    two data rows of 表五六's continuation page in <thead>, and they stay as they are."""
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
    if flatten and p.head > 1 and p.head_spans:
        grid = [_flat_header(grid[: p.head]), *grid[p.head :]]
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


def glm_to_markdown(raw: str, *, flatten: bool = False) -> str:
    """glm-ocr through ollama answers in markdown; HTML gets expanded, anything else is kept
    as it came (plain text means the model saw no table -- that is the result, not a bug)."""
    if "<table" in raw:
        return html_to_markdown(raw, flatten=flatten)
    return raw.strip()


def norm(s: str) -> str:
    """For comparing cell text across sources: NFKC, no spaces, ASCII punctuation."""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", "", s)
    return s.replace("，", ",").replace("。", ".").replace("（", "(").replace("）", ")")
