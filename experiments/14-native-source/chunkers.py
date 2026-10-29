"""01's chunkers, unchanged, plus what the gazettes need: article headers written with Chinese
numerals on the same line as the text (第十四條之一　漁船…, 第 六 條 學校…), and a law-name
prefix on every chunk (07's title chain, for a corpus of thirty look-alike regulations)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_HEAD = re.compile(r"^第 (\d+) 條$", re.M)
_CHAP = re.compile(r"^\s*第 ([一二三四五六七八九十]+) 章\s*(\S.*?)\s*$", re.M)
_SECT = re.compile(r"^\s*第 ([一二三四五六七八九十]+) 節\s*(\S.*?)\s*$", re.M)
_NUMERIC = re.compile(r"^[0-9][0-9.,/\-~～%]*$")
_CN_VALUE = {ch: i for i, ch in enumerate("零一二三四五六七八九")} | {"〇": 0}
_CN = "零〇一二三四五六七八九十百"
SEPS = ["\n\n", "\n", ". ", " ", ""]
_POINT = re.compile(r"^([一二三四五六七八九十百]+)、", re.M)
_HEAD_CN = re.compile(
    rf"^第\s*([{_CN}](?:\s*[{_CN}])*)\s*條(?:\s*之\s*([{_CN}]+))?(?=[\s　（(]|$)", re.M
)
_MARK_CN = re.compile(rf"^第\s*[{_CN}]+\s*[章節]")


def cn_to_int(s: str) -> int:
    """Chinese numeral (零 to 九百九十九, 十 and 百 as place words) to int."""
    s = s.replace(" ", "")
    total = 0
    num = 0
    for ch in s:
        if ch == "百":
            total += (num or 1) * 100
            num = 0
        elif ch == "十":
            total += (num or 1) * 10
            num = 0
        else:
            num = _CN_VALUE[ch]
    return total + num


def int_to_cn(n: int) -> str:
    """int (1–999) to the numeral the gazettes print: 十四, 二十一, 一百零八, 一百二十."""
    d = "零一二三四五六七八九"
    if n < 10:
        return d[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return (d[tens] if tens > 1 else "") + "十" + (d[ones] if ones else "")
    hundreds, rest = divmod(n, 100)
    out = d[hundreds] + "百"
    if rest == 0:
        return out
    if rest < 10:
        return out + "零" + d[rest]
    if rest < 20:
        return out + "一十" + (d[rest % 10] if rest % 10 else "")
    return out + int_to_cn(rest)


@dataclass
class Article:
    num: int
    chapter: str
    section: str
    body: str
    start: int  # span in the source text, header line included
    end: int
    label: str = ""  # "14-1" for 第十四條之一; defaults to str(num)

    def __post_init__(self) -> None:
        if not self.label:
            self.label = str(self.num)


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


def parse_articles_cn(text: str) -> list[Article]:
    """Articles whose header is 第<Chinese numeral>條[之N] at the start of a line, with the text
    continuing on the same line. Chapter / section lines are dropped from bodies, not tracked."""
    heads = list(_HEAD_CN.finditer(text))
    out: list[Article] = []
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        raw = text[h.end() : end]
        body = "\n".join(
            ln.strip() for ln in raw.splitlines() if ln.strip() and not _MARK_CN.match(ln.strip())
        ).strip()
        num = cn_to_int(h.group(1))
        label = f"{num}-{cn_to_int(h.group(2))}" if h.group(2) else str(num)
        out.append(Article(num, "", "", body, h.start(), end, label))
    logger.debug("articles parsed (cn)", extra={"ctx": {"n": len(out)}})
    return out


def parse_points(text: str) -> list[Article]:
    """行政規則 points: a 中文數字、 at the very start of a line (no indent) opens a point; 款
    （一） and 目 1. lines stay in the point's body. Returns Article objects (num = point number,
    label = str(num)) so the chunkers below can treat points like articles."""
    heads = list(_POINT.finditer(text))
    out: list[Article] = []
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        raw = text[h.end() : end]
        body = "\n".join(ln.strip() for ln in raw.splitlines() if ln.strip()).strip()
        num = cn_to_int(h.group(1))
        out.append(Article(num, "", "", body, h.start(), end, str(num)))
    logger.debug("points parsed", extra={"ctx": {"n": len(out)}})
    return out


def points_chunks(text: str, *, max_len: int = 1000) -> list[Chunk]:
    """Front matter (before the first point) as fixed chunks labelled "pre", then one chunk per
    point headed 第 N 點; bodies over max_len split at line boundaries (款 are one per line) with
    the header repeated. No point -> no chunk (the caller falls back)."""
    pts = parse_points(text)
    if not pts:
        return []
    out: list[Chunk] = []
    pre = text[: pts[0].start]
    if pre.strip():
        front = fixed_chunks(pre, size=max_len, overlap=0, separators=SEPS, spans=[])
        out += [Chunk(c.text, ["pre"], c.start, c.end, dict(c.meta)) for c in front]
    for a in pts:
        header = f"第 {a.label} 點\n"
        pieces: list[str] = []
        buf: list[str] = []
        for ln in a.body.splitlines():
            if buf and sum(len(x) + 1 for x in buf) + len(ln) > max_len:
                pieces.append("\n".join(buf))
                buf = []
            buf.append(ln)
        pieces.append("\n".join(buf))
        for piece in pieces:
            out.append(Chunk(header + piece, [a.label], a.start, a.end, {"point": a.label}))
    return out


def with_prefix(chunks: list[Chunk], name: str) -> list[Chunk]:
    """Every chunk restated with the law's name on its first line (07's title chain)."""
    return [
        Chunk(f"《{name}》\n{c.text}", list(c.labels), c.start, c.end, dict(c.meta)) for c in chunks
    ]


def article_chunks(articles: list[Article], *, max_len: int = 1000) -> list[Chunk]:
    """One chunk per 條, header-prefixed. Bodies over max_len are split at line boundaries
    (款/項 are one per line in the PDF text) and every piece repeats the header."""
    out: list[Chunk] = []
    for a in articles:
        header = f"{a.chapter} / {a.section} / 第 {a.label} 條\n"
        pieces: list[str] = []
        buf: list[str] = []
        for ln in a.body.splitlines():
            if buf and sum(len(x) + 1 for x in buf) + len(ln) > max_len:
                pieces.append("\n".join(buf))
                buf = []
            buf.append(ln)
        pieces.append("\n".join(buf))
        for p in pieces:
            out.append(Chunk(header + p, [a.label], a.start, a.end, {"article": a.label}))
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


def fixed_chunks(
    text: str, *, size: int, overlap: int, separators: list[str], spans: list[tuple[int, int, str]]
) -> list[Chunk]:
    """RecursiveCharacterTextSplitter-style: split on the first separator present, recurse on
    pieces still too long, then merge greedily up to `size` keeping `overlap` chars of tail.
    Labels = every span that intersects the chunk."""
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
        labels = [lab for a, b, lab in spans if a < e2 and b > s2]
        out.append(Chunk(stripped, labels, s2, e2))
    logger.debug("fixed chunks", extra={"ctx": {"size": size, "overlap": overlap, "n": len(out)}})
    return out


def table_whole_chunks(tables: dict[str, str]) -> list[Chunk]:
    return [Chunk(text, [tid], meta={"table": tid}) for tid, text in tables.items()]


def _is_row(line: str) -> bool:
    toks = line.split()
    return len(toks) >= 2 and bool(_NUMERIC.match(toks[-1]))


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
