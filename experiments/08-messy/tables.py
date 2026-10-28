"""PDF table -> grid with exact spans -> markdown / html / kv text. Copied from 04 (experiments
do not share code); the only change is load_table(table_settings=...).


Spans come from pdfplumber's cell geometry (a merged cell is one bbox covering several column
and row edges), not from guessing which empty neighbour an empty cell belongs to. Two things the
geometry cannot tell are handled by rules and documented in the README:
  * header depth: the last of the first four rows that holds a colspan, extended by rowspans
    that start inside the header (override per table when the rule is wrong, e.g. 表四○);
  * rows the PDF does not rule apart (表二五～二…五): one cell holds many lines, so a data row
    whose cells carry >= 3 lines each is exploded into one row per line.
"""

from __future__ import annotations

import html as html_mod
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

logger = logging.getLogger(__name__)

_NOTE = re.compile(r"^\s*註")
_TITLE = re.compile(r"^表[^\s]+")
_NUMERIC_LINE = re.compile(r"^[\d.,/\-–\s]+$")


@dataclass
class Cell:
    text: str
    rowspan: int = 1
    colspan: int = 1


@dataclass
class Table:
    title: str
    cells: list[list[Cell | None]]  # None = covered by a span whose origin is up/left
    header_rows: int
    notes: str = ""
    source: str = ""
    _filled: list[list[str]] | None = field(default=None, repr=False)

    # -- views -------------------------------------------------------------------------------
    def n_cols(self) -> int:
        return len(self.cells[0]) if self.cells else 0

    def filled(self) -> list[list[str]]:
        """Text grid with every covered position carrying its origin's text."""
        if self._filled is None:
            n = self.n_cols()
            out = [[""] * n for _ in self.cells]
            for i, row in enumerate(self.cells):
                for j, c in enumerate(row):
                    if c is None:
                        continue
                    for di in range(c.rowspan):
                        for dj in range(c.colspan):
                            if i + di < len(out) and j + dj < n:
                                out[i + di][j + dj] = c.text
            self._filled = out
        return self._filled

    def data_rows(self) -> list[list[str]]:
        return self.filled()[self.header_rows :]

    def column_names(self) -> list[str]:
        names = []
        for j in range(self.n_cols()):
            parts: list[str] = []
            for i in range(self.header_rows):
                t = self.filled()[i][j]
                if t and (not parts or parts[-1] != t):
                    parts.append(t)
            names.append("／".join(parts))
        return names


# -- extraction -------------------------------------------------------------------------------


def _one_line(s: str | None) -> str:
    return re.sub(r"\s*\n\s*", "", (s or "").strip())


def _edges(bboxes: list[tuple], axis: int) -> list[float]:
    vals = sorted({round(b[axis], 1) for b in bboxes} | {round(b[axis + 2], 1) for b in bboxes})
    return vals


def _span(lo: float, hi: float, edges: list[float]) -> int:
    inside = [e for e in edges if lo + 0.5 < e < hi - 0.5]
    return len(inside) + 1


def _page_cells(table: pdfplumber.table.Table) -> list[list[Cell | None]]:
    """Cells of one page's table, spans derived from bbox vs the table's row/column edges."""
    texts = table.extract()
    xs = _edges(table.cells, 0)
    ys = _edges(table.cells, 1)
    out: list[list[Cell | None]] = []
    for i, row in enumerate(table.rows):
        line: list[Cell | None] = []
        for j, bbox in enumerate(row.cells):
            if bbox is None:
                line.append(None)
                continue
            x0, top, x1, bottom = bbox
            line.append(
                Cell(
                    text=(texts[i][j] or "").strip(),
                    rowspan=_span(top, bottom, ys),
                    colspan=_span(x0, x1, xs),
                )
            )
        out.append(line)
    return out


def _strip_notes(cells: list[list[Cell | None]]) -> tuple[list[list[Cell | None]], str]:
    notes = []
    kept = []
    for row in cells:
        first = next((c for c in row if c is not None), None)
        if first is not None and _NOTE.match(first.text):
            notes.append(_one_line(first.text))
        else:
            kept.append(row)
    return kept, "\n".join(notes)


def _guess_header_rows(cells: list[list[Cell | None]]) -> int:
    h = 1
    for i in range(min(4, len(cells))):
        if any(c is not None and c.colspan > 1 for c in cells[i]):
            h = i + 1
    # a rowspan that starts inside the header pulls the rows it covers into the header
    changed = True
    while changed:
        changed = False
        for i in range(h):
            for c in cells[i]:
                if c is not None and i + c.rowspan > h:
                    h = i + c.rowspan
                    changed = True
    return h


def _explode(row: list[Cell | None]) -> list[list[Cell | None]]:
    """A data row whose cells hold several lines each becomes one row per line."""
    parts = [(c.text.split("\n") if c is not None else []) for c in row]
    multi = [p for p in parts if len(p) >= 3]
    if len(multi) < 2:
        return [row]
    lines = [ln.strip() for p in multi for ln in p if ln.strip()]
    if sum(bool(_NUMERIC_LINE.match(ln)) for ln in lines) < 0.8 * len(lines):
        return [row]  # prose cells (表九二): the lines are wrapped text, not stacked rows
    n = max(len(p) for p in parts)
    rows: list[list[Cell | None]] = []
    for k in range(n):
        line: list[Cell | None] = []
        for j, (c, p) in enumerate(zip(row, parts, strict=True)):
            if c is None:
                line.append(None)
            elif j == 0 and len(p) < n:  # a label spanning the exploded rows
                line.append(Cell("".join(p), rowspan=n) if k == 0 else None)
            else:
                line.append(Cell(p[k].strip() if k < len(p) else "", colspan=c.colspan))
        rows.append(line)
    return rows


def _drop_columns(cells: list[list[Cell | None]], keep: list[int]) -> list[list[Cell | None]]:
    """Remove the columns not in `keep`; origins whose span is cut lose the dropped part."""
    n = len(cells[0]) if cells else 0
    new_rows: list[list[Cell | None]] = []
    for row in cells:
        line: list[Cell | None] = []
        j = 0
        while j < n:
            c = row[j]
            if c is None:
                if j in keep:
                    line.append(None)
                j += 1
                continue
            covered = [jj for jj in range(j, j + c.colspan) if jj in keep]
            if covered:
                line.append(Cell(c.text, c.rowspan, len(covered)))
                line.extend([None] * (len(covered) - 1))
            j += c.colspan
        new_rows.append(line)
    return new_rows


def _drop_degenerate_columns(cells: list[list[Cell | None]]) -> list[list[Cell | None]]:
    """A column that owns no cell below the first row is half of a diagonally split corner
    (表八六～三): every later row spans across it. Drop it, header text included."""
    n = len(cells[0]) if cells else 0
    keep = [j for j in range(n) if j == 0 or any(row[j] is not None for row in cells[1:])]
    return cells if len(keep) == n else _drop_columns(cells, keep)


def _drop_empty_columns(t: Table) -> Table:
    """A column with no data in any data row is removed, together with its header text."""
    data = t.filled()[t.header_rows :]
    n = t.n_cols()
    keep = [j for j in range(n) if any(r[j] for r in data)] if data else list(range(n))
    if len(keep) == n:
        return t
    return Table(t.title, _drop_columns(t.cells, keep), t.header_rows, t.notes, t.source)


def load_table(
    path: Path | str, *, header_rows: int | None = None, table_settings: dict | None = None
) -> Table:
    """08: `table_settings` is passed to pdfplumber (None = the default lines strategy; 08's
    borderless variant uses {"vertical_strategy": "text", "horizontal_strategy": "text"})."""
    path = Path(path)
    cells: list[list[Cell | None]] = []
    title = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            found = page.find_tables(table_settings) if table_settings else page.find_tables()
            if not found:
                continue
            if not title:
                for ln in (page.extract_text() or "").splitlines():
                    if _TITLE.match(ln.strip()):
                        title = ln.strip()
                        break
            cells.extend(_page_cells(found[0]))
    cells, notes = _strip_notes(cells)
    cells = _drop_degenerate_columns(cells)
    h = header_rows if header_rows is not None else _guess_header_rows(cells)
    for row in cells[:h]:
        for c in row:
            if c is not None:
                c.text = _one_line(c.text)
    body: list[list[Cell | None]] = []
    for row in cells[h:]:
        body.extend(_explode(row))
    for row in body:
        for c in row:
            if c is not None:
                c.text = _one_line(c.text)
    t = _drop_empty_columns(Table(title, cells[:h] + body, h, notes, str(path)))
    logger.debug(
        "table loaded",
        extra={
            "ctx": {
                "path": path.name,
                "title": title,
                "header_rows": h,
                "rows": len(t.cells),
                "cols": t.n_cols(),
            }
        },
    )
    return t


# -- renderings -------------------------------------------------------------------------------


def to_markdown(t: Table) -> str:
    names = t.column_names()
    lines = [t.title, "| " + " | ".join(names) + " |", "|" + "---|" * len(names)]
    for r in t.data_rows():
        lines.append("| " + " | ".join(r) + " |")
    if t.notes:
        lines.append(t.notes)
    return "\n".join(lines)


def to_kv(t: Table) -> str:
    names = t.column_names()
    lines = [t.title]
    for r in t.data_rows():
        lines.append("；".join(f"{n}={v}" for n, v in zip(names, r, strict=True) if v))
    if t.notes:
        lines.append(t.notes)
    return "\n".join(lines)


def to_html(t: Table) -> str:
    esc = html_mod.escape
    out = ["<table>", f"<caption>{esc(t.title)}</caption>"]
    for i, row in enumerate(t.cells):
        tag = "th" if i < t.header_rows else "td"
        tds = []
        for c in row:
            if c is None:
                continue
            attrs = ""
            if c.rowspan > 1:
                attrs += f' rowspan="{c.rowspan}"'
            if c.colspan > 1:
                attrs += f' colspan="{c.colspan}"'
            tds.append(f"<{tag}{attrs}>{esc(c.text)}</{tag}>")
        out.append("<tr>" + "".join(tds) + "</tr>")
    out.append("</table>")
    if t.notes:
        out.append(f"<p>{esc(t.notes)}</p>")
    return "\n".join(out)


# -- 04: range cells -> bound columns --------------------------------------------------------

_N = r"(\d[\d,]*(?:\.\d+)?)"
_RANGE_PATTERNS = [
    (re.compile(rf"^(?:超過\s*)?{_N}\s*[-–~～至]\s*{_N}$"), lambda m: (_num(m[1]), _num(m[2]))),
    (re.compile(rf"^{_N}\s*以下$"), lambda m: (None, _num(m[1]))),
    (re.compile(rf"^(?:超過\s*)?{_N}\s*以上$"), lambda m: (_num(m[1]), None)),
    (re.compile(rf"^超過\s*{_N}$"), lambda m: (_num(m[1]), None)),
    (re.compile(rf"^{_N}$"), lambda m: (_num(m[1]), _num(m[1]))),
]


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def parse_range(text: str) -> tuple[float | None, float | None] | None:
    """'38 - 50' -> (38, 50); '30以下' -> (None, 30); '41以上' / '超過120,000' -> (x, None);
    a bare number -> (x, x); anything else (text, '15或20', '7/0.8', 'l00') -> None."""
    s = text.strip()
    for pat, fn in _RANGE_PATTERNS:
        m = pat.match(s)
        if m:
            return fn(m)
    return None


def _fmt(x: float | None) -> str:
    if x is None:
        return ""
    return str(int(x)) if x == int(x) else str(x)


def expand_ranges(t: Table) -> Table:
    """Insert '<name>／下限' and '<name>／上限' after every column whose data cells are mostly
    ranges (and not all single points). Header levels are flattened into one row; 04 only renders
    markdown, so the span information is not needed."""
    names = t.column_names()
    data = t.data_rows()
    expand: list[int] = []
    for j in range(len(names)):
        cells = [r[j] for r in data if r[j]]
        parsed = [parse_range(c) for c in cells]
        ok = [p for p in parsed if p is not None]
        if (
            cells
            and len(ok) >= 0.5 * len(cells)
            and any(p[0] is None or p[1] is None or p[0] != p[1] for p in ok)
        ):
            expand.append(j)
    if not expand:
        return t
    new_names: list[str] = []
    for j, n in enumerate(names):
        new_names.append(n)
        if j in expand:
            new_names += [f"{n}／下限", f"{n}／上限"]
    new_rows: list[list[Cell | None]] = [[Cell(n) for n in new_names]]
    for r in data:
        line: list[Cell | None] = []
        for j, v in enumerate(r):
            line.append(Cell(v))
            if j in expand:
                p = parse_range(v) or (None, None)
                line += [Cell(_fmt(p[0])), Cell(_fmt(p[1]))]
        new_rows.append(line)
    return Table(t.title, new_rows, 1, t.notes, t.source)


# -- 04: flat JSON round trip (run.py must work on a machine without the PDFs) ----------------


def table_to_json(t: Table) -> dict:
    return {"title": t.title, "names": t.column_names(), "rows": t.data_rows(), "notes": t.notes}


def table_from_json(d: dict) -> Table:
    cells: list[list[Cell | None]] = [[Cell(n) for n in d["names"]]]
    cells += [[Cell(v) for v in r] for r in d["rows"]]
    return Table(d["title"], cells, 1, d.get("notes", ""))
