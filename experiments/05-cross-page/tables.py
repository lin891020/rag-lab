"""PDF table -> grid with exact spans -> markdown text, with tables that continue over pages.

Copied from 04 (03's loader) and extended: `load_pages` keeps each page's table as its own grid
(with the cell bboxes), `is_continuation` is the rule that says page N+1's table continues page
N's (docs/research/2026-10-29-cross-page-tables.md, section 最便宜的作法), and `merge_pages`
aligns the pages on the union of their column edges, carries a cut rowspan label over the page
break and joins a cell whose text was split by it. `load_page` is the naive per-page reading
(what a page-chunked pipeline sees).

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
    pages: list[int] = field(default_factory=list)
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


@dataclass
class PageGrid:
    """One page's table as pdfplumber found it, with the cell boxes kept."""

    page_no: int
    page_height: float
    bbox: tuple[float, float, float, float]
    rows: list[list[tuple[tuple[float, float, float, float], str] | None]]
    x_edges: list[float]
    y_edges: list[float]
    has_title: bool
    words_above: int  # words on the page above the table (a title counts)
    words_below: int  # words on the page below the table


BOTTOM_TH = 0.8  # previous table must reach this fraction of the page height
TOP_TH = 0.2  # next table must start above this fraction
EDGE_TOL = 3.0  # pt: two column edges are "the same"
EDGE_MATCH = 0.6  # fraction of the next page's edges that must match
WIDTH_TOL = 0.1  # relative width difference allowed


def load_pages(path: Path | str) -> list[PageGrid]:
    """One grid per page that has a table (pages without one are skipped)."""
    out = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            found = page.find_tables()
            if not found:
                continue
            t = found[0]
            words = page.extract_words()
            has_title = any(
                _TITLE.match(ln.strip()) for ln in (page.extract_text() or "").splitlines()
            )
            rows = []
            for row in t.rows:
                line = []
                texts = [page.crop(b).extract_text() if b else None for b in row.cells]
                for b, tx in zip(row.cells, texts, strict=True):
                    line.append(None if b is None else (b, (tx or "").strip()))
                rows.append(line)
            out.append(
                PageGrid(
                    page_no=page.page_number - 1,
                    page_height=float(page.height),
                    bbox=tuple(float(v) for v in t.bbox),  # type: ignore[arg-type]
                    rows=rows,
                    x_edges=_edges(t.cells, 0),
                    y_edges=_edges(t.cells, 1),
                    has_title=has_title,
                    words_above=sum(w["bottom"] <= t.bbox[1] + 1 for w in words),
                    words_below=sum(w["top"] >= t.bbox[3] - 1 for w in words),
                )
            )
    return out


def _edges_match(prev: list[float], cur: list[float]) -> float:
    return sum(any(abs(e - p) <= EDGE_TOL for p in prev) for e in cur) / max(1, len(cur))


def is_continuation(prev: PageGrid, cur: PageGrid) -> bool:
    """Rule from the survey: previous table reaches the page bottom, this one starts at the top
    with nothing (no title) above it, same width, most column edges line up."""
    if cur.page_no != prev.page_no + 1:
        return False
    if prev.bbox[3] < BOTTOM_TH * prev.page_height or prev.words_below:
        return False
    if cur.bbox[1] > TOP_TH * cur.page_height or cur.words_above or cur.has_title:
        return False
    w_prev, w_cur = prev.bbox[2] - prev.bbox[0], cur.bbox[2] - cur.bbox[0]
    if abs(w_prev - w_cur) > WIDTH_TOL * max(w_prev, w_cur):
        return False
    return _edges_match(prev.x_edges, cur.x_edges) >= EDGE_MATCH


def _union_edges(grids: list[PageGrid]) -> list[float]:
    xs = sorted(e for g in grids for e in g.x_edges)
    out: list[float] = []
    for x in xs:
        if not out or x - out[-1] > EDGE_TOL:
            out.append(x)
    return out


def _grid_cells(g: PageGrid, union: list[float]) -> list[list[Cell | None]]:
    """The page's rows as Cells whose colspans are measured against the union of edges."""
    out: list[list[Cell | None]] = []
    for row in g.rows:
        line: list[Cell | None] = []
        covered_to = g.x_edges[0]
        for j, item in enumerate(row):
            x_lo, x_hi = g.x_edges[j], g.x_edges[j + 1]
            if item is not None:
                (x0, top, x1, bottom), text = item
                span = _span(x0, x1, union)
                line.append(Cell(text, rowspan=_span(top, bottom, g.y_edges), colspan=span))
                line.extend([None] * (span - 1))
                covered_to = x1
            elif x_hi > covered_to + EDGE_TOL:  # covered from above, not from the left
                line.extend([None] * _span(x_lo, x_hi, union))
                covered_to = x_hi
        if len(line) != len(union) - 1:
            logger.warning(
                "row width mismatch",
                extra={"ctx": {"page": g.page_no, "got": len(line), "want": len(union) - 1}},
            )
            line = (line + [None] * (len(union) - 1))[: len(union) - 1]
        out.append(line)
    return out


def _origin_at(row: list[Cell | None], j: int) -> int:
    while j > 0 and row[j] is None:
        j -= 1
    return j


def merge_pages(grids: list[PageGrid]) -> tuple[list[list[Cell | None]], list[int]]:
    """Cells of the first grid plus every grid the rule accepts as its continuation; returns
    (cells, page numbers used). Two page-break repairs, both only on the first row of a
    continuation: a row with a single non-empty cell is the tail of the previous row's cell
    (joined back); leading empty cells take the label a cut rowspan carried from above."""
    used = [grids[0]]
    for g in grids[1:]:
        if is_continuation(used[-1], g):
            used.append(g)
        else:
            break
    union = _union_edges(used)
    cells = _grid_cells(used[0], union)
    for g in used[1:]:
        rows = _grid_cells(g, union)
        first = rows[0]
        origins = [(j, c) for j, c in enumerate(first) if c is not None]
        non_empty = [(j, c) for j, c in origins if c.text]
        if len(non_empty) == 1 and len(origins) > 1 and cells:
            j, c = non_empty[0]
            k = _origin_at(cells[-1], j)
            tgt = cells[-1][k]
            if tgt is not None:
                tgt.text = _one_line(tgt.text) + _one_line(c.text)
                rows = rows[1:]
                logger.debug("joined split cell", extra={"ctx": {"page": g.page_no, "col": j}})
        if rows and cells:
            last_filled = Table("", cells, 0).filled()[-1]
            for j, c in enumerate(rows[0]):
                if c is None or c.text:
                    break
                c.text = last_filled[j]
                logger.debug("carried label", extra={"ctx": {"page": g.page_no, "col": j}})
        cells.extend(rows)
    return cells, [g.page_no for g in used]


def _finish(
    title: str,
    cells: list[list[Cell | None]],
    source: str,
    header_rows: int | None,
    pages: list[int],
) -> Table:
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
    t = _drop_empty_columns(Table(title, cells[:h] + body, h, notes, source, pages=pages))
    logger.debug(
        "table loaded",
        extra={
            "ctx": {
                "path": Path(source).name,
                "title": title,
                "header_rows": h,
                "rows": len(t.cells),
                "cols": t.n_cols(),
                "pages": pages,
            }
        },
    )
    return t


def _title_of(path: Path, page_no: int) -> str:
    with pdfplumber.open(path) as pdf:
        for ln in (pdf.pages[page_no].extract_text() or "").splitlines():
            if _TITLE.match(ln.strip()):
                return ln.strip()
    return ""


def load_table(path: Path | str, *, header_rows: int | None = None, merge: bool = True) -> Table:
    """The whole table: first page plus the pages the rule accepts as continuations."""
    path = Path(path)
    grids = load_pages(path)
    if merge:
        cells, pages = merge_pages(grids)
    else:
        cells, pages = _grid_cells(grids[0], grids[0].x_edges), [grids[0].page_no]
    return _finish(_title_of(path, grids[0].page_no), cells, str(path), header_rows, pages)


def load_page(path: Path | str, page_no: int, *, header_rows: int | None = None) -> Table:
    """One page on its own, as a page-chunked pipeline reads it: no merge, the title only if the
    page carries one, and (default) the first row taken as the header."""
    path = Path(path)
    g = next(g for g in load_pages(path) if g.page_no == page_no)
    cells = _grid_cells(g, g.x_edges)
    title = _title_of(path, page_no) if g.has_title else ""
    return _finish(title, cells, str(path), 1 if header_rows is None else header_rows, [page_no])


# -- renderings -------------------------------------------------------------------------------


def to_markdown(t: Table) -> str:
    names = t.column_names()
    lines = [t.title] if t.title else []
    lines += ["| " + " | ".join(names) + " |", "|" + "---|" * len(names)]
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


def table_to_json(t: Table) -> dict:
    return {"title": t.title, "names": t.column_names(), "rows": t.data_rows(), "notes": t.notes}


def table_from_json(d: dict) -> Table:
    cells: list[list[Cell | None]] = [[Cell(n) for n in d["names"]]]
    cells += [[Cell(v) for v in r] for r in d["rows"]]
    return Table(d["title"], cells, 1, d.get("notes", ""))
