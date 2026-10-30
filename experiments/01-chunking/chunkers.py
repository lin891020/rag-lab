"""The chunking strategies under test. Every chunk carries the labels (條號 or 表號) it covers."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_HEAD = re.compile(r"^第 (\d+) 條$", re.M)
_CHAP = re.compile(r"^\s*第 ([一二三四五六七八九十]+) 章\s*(\S.*?)\s*$", re.M)
_SECT = re.compile(r"^\s*第 ([一二三四五六七八九十]+) 節\s*(\S.*?)\s*$", re.M)
_NUMERIC = re.compile(r"^[0-9][0-9.,/\-~～%]*$")


@dataclass
class Article:
    num: int
    chapter: str
    section: str
    body: str
    start: int  # span in the source text, header line included
    end: int


@dataclass
class Chunk:
    text: str
    labels: list[str]
    start: int = 0
    end: int = 0
    meta: dict = field(default_factory=dict)


def parse_articles(text: str) -> list[Article]:
    heads = list(_HEAD.finditer(text))
    marks = sorted(
        [(m.start(), "chapter", f"第{m.group(1)}章 {m.group(2)}") for m in _CHAP.finditer(text)]
        + [(m.start(), "section", f"第{m.group(1)}節 {m.group(2)}") for m in _SECT.finditer(text)]
    )
    out: list[Article] = []
    chapter = section = ""
    mi = 0
    for i, h in enumerate(heads):
        while mi < len(marks) and marks[mi][0] < h.start():
            if marks[mi][1] == "chapter":
                chapter, section = marks[mi][2], ""
            else:
                section = marks[mi][2]
            mi += 1
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        raw = text[h.end() : end]
        body = "\n".join(
            ln for ln in raw.splitlines() if ln.strip() and not (_CHAP.match(ln) or _SECT.match(ln))
        ).strip()
        out.append(Article(int(h.group(1)), chapter, section, body, h.start(), end))
    logger.debug("articles parsed", extra={"ctx": {"n": len(out)}})
    return out


def article_chunks(articles: list[Article], *, max_len: int = 1000) -> list[Chunk]:
    """One chunk per 條, header-prefixed. Bodies over max_len are split at line boundaries
    (款/項 are one per line in the PDF text) and every piece repeats the header."""
    out: list[Chunk] = []
    for a in articles:
        header = f"{a.chapter} / {a.section} / 第 {a.num} 條\n"
        pieces: list[str] = []
        buf: list[str] = []
        for ln in a.body.splitlines():
            if buf and sum(len(x) + 1 for x in buf) + len(ln) > max_len:
                pieces.append("\n".join(buf))
                buf = []
            buf.append(ln)
        pieces.append("\n".join(buf))
        for p in pieces:
            out.append(Chunk(header + p, [str(a.num)], a.start, a.end, {"article": a.num}))
    return out


def _pieces(text: str, s: int, e: int, size: int, seps: list[str]) -> list[tuple[int, int]]:
    """Recursive split into spans no longer than size (unless indivisible), separator kept at the end."""
    if e - s <= size:
        return [(s, e)]
    for i, sep in enumerate(seps):
        if sep == "":
            return [(j, min(j + size, e)) for j in range(s, e, size)]
        seg = text[s:e]
        if sep not in seg:
            continue
        out: list[tuple[int, int]] = []
        cur = s
        for part in seg.split(sep):
            ps, pe = cur, min(cur + len(part) + len(sep), e)
            cur = pe
            if pe <= ps:
                continue
            out.extend(_pieces(text, ps, pe, size, seps[i + 1 :]) if pe - ps > size else [(ps, pe)])
        return out
    return [(s, e)]


MIN_SPAN_CHARS = 100  # a chunk carries a span's label only when it holds this much of it ...


def span_labels(s: int, e: int, spans: list[tuple[int, int, str]]) -> list[str]:
    """Labels of the spans the chunk [s, e) really holds: at least min(100 chars, half the
    span). Touching a span by a few characters used to count (the 2026-10-29 audit: a fifth of
    bm25's fixed269 "hits" held no answer); half of a short span is the floor so that a 40-char
    article cut in two still labels both halves."""
    out = []
    for a, b, lab in spans:
        held = min(b, e) - max(a, s)
        if held > 0 and held >= min(MIN_SPAN_CHARS, -(-(b - a) // 2)):
            out.append(lab)
    return out


def fixed_chunks(
    text: str, *, size: int, overlap: int, separators: list[str], spans: list[tuple[int, int, str]]
) -> list[Chunk]:
    """RecursiveCharacterTextSplitter-style: split on the first separator present, recurse on
    pieces still too long, then merge greedily up to `size` keeping `overlap` chars of tail.
    Labels = every span the chunk holds a real piece of (span_labels)."""
    pieces = _pieces(text, 0, len(text), size, separators)
    bounds: list[tuple[int, int]] = []
    buf: list[tuple[int, int]] = []
    total = 0
    for ps, pe in pieces:
        n = pe - ps
        if buf and total + n > size:
            bounds.append((buf[0][0], buf[-1][1]))
            while buf and (total > overlap or (total + n > size and total > 0)):
                total -= buf[0][1] - buf[0][0]
                buf.pop(0)
        buf.append((ps, pe))
        total += n
    if buf:
        bounds.append((buf[0][0], buf[-1][1]))
    out: list[Chunk] = []
    for s, e in bounds:
        raw = text[s:e]
        stripped = raw.strip()
        if not stripped:
            continue
        s2 = s + (len(raw) - len(raw.lstrip()))
        e2 = s2 + len(stripped)
        out.append(Chunk(stripped, span_labels(s2, e2, spans), s2, e2))
    logger.debug("fixed chunks", extra={"ctx": {"size": size, "overlap": overlap, "n": len(out)}})
    return out


def table_whole_chunks(tables: dict[str, str]) -> list[Chunk]:
    return [Chunk(text, [tid], meta={"table": tid}) for tid, text in tables.items()]


_HAS_DIGIT = re.compile(r"[0-9]")


def _is_row(line: str) -> bool:
    """A data row: two or more tokens and a number somewhere in the last one. The 2026-10-29
    audit found 7 tables with no row under the old rule (last token had to be purely numeric:
    值 like 25以下 or the OCR'd l0以下 never qualified), among them t15's gold 表九二."""
    toks = line.split()
    return len(toks) >= 2 and bool(_HAS_DIGIT.search(toks[-1]))


def table_row_chunks(tables: dict[str, str]) -> list[Chunk]:
    """One chunk per data row, prefixed by the title and every line above the first data row.
    Header ends at the first all-numeric line, or (tables with text keys) the first line whose
    last token is numeric. Lines after that which are not rows (stray labels, 註) are dropped."""
    out: list[Chunk] = []
    for tid, text in tables.items():
        lines = text.splitlines()
        first = next(
            (
                i
                for i, ln in enumerate(lines)
                if all(_NUMERIC.match(t) for t in ln.split()) and len(ln.split()) >= 2
            ),
            None,
        )
        if first is None:
            first = next((i for i, ln in enumerate(lines) if _is_row(ln)), len(lines))
        header = "\n".join(lines[:first])
        for ln in lines[first:]:
            if ln.startswith("註"):
                break
            if _is_row(ln):
                out.append(Chunk(header + "\n" + ln, [tid], meta={"table": tid}))
    return out


def table_flat_fixed_chunks(
    tables: dict[str, str], *, size: int, overlap: int, separators: list[str]
) -> list[Chunk]:
    """Tables glued into one text and cut with the fixed splitter, as a naive pipeline would."""
    spans: list[tuple[int, int, str]] = []
    parts: list[str] = []
    pos = 0
    for tid, text in tables.items():
        spans.append((pos, pos + len(text), tid))
        parts.append(text)
        pos += len(text) + 2
    return fixed_chunks(
        "\n\n".join(parts), size=size, overlap=overlap, separators=separators, spans=spans
    )
