"""One version of 用戶用電設備裝置規則 from the national law database page -> text -> article
units. 28's laws.py trimmed: no tables, no list of laws; `law_units` takes the version prefix.

The page is turned into text by 14's `html_text` (copied below: tags off, one paragraph per
line, site chrome dropped). The page writes 「本條文有附件 第 7 條」 when an article has an
attachment; the prefix is dropped so every article head stands on its own line. Article numbers
keep their branch (第 13-1 條). Units: header 「章 / 節 / 第 N 條」, bodies over max_len split at
line boundaries, every piece repeats the header, labelled `<prefix>:<num>` (old:16, old:13-1).
"""

from __future__ import annotations

import html as htmllib
import logging
import re
from dataclasses import dataclass

from chunkers import Chunk

logger = logging.getLogger(__name__)

BASE_NAME = "用戶用電設備裝置規則"

_HEAD = re.compile(r"^第 (\d+(?:-\d+)?) 條$", re.M)
_CHAP = re.compile(r"^\s*第 ([一二三四五六七八九十百]+) 章\s*(\S.*?)\s*$", re.M)
_SECT = re.compile(r"^\s*第 ([一二三四五六七八九十百]+) 節\s*(\S.*?)\s*$", re.M)
_PART = re.compile(
    r"^\s*第 ([一二三四五六七八九十百]+) 編\s*(\S.*?)\s*$", re.M
)  # not in this law; kept
_SUB = re.compile(
    r"^\s*第 ([一二三四五六七八九十百]+) 款\s*(\S.*?)\s*$", re.M
)  # the old version's 款 headings
_SOURCE = re.compile(r"^資料來源：.*$\n?", re.M)  # the page's last line
_ATTACHED = re.compile(r"^本條文有附件\s*(?=第 )", re.M)
_NAME = re.compile(r'id="hlLawName"[^>]*>([^<]+)<')
_AMENDED = re.compile(r"修正日期：</th>\s*<td>([^<]+)<")

# -- 14's html_text, copied (experiments/14-native-source/extract.py) ---------------------------
_BLOCK = re.compile(r"</?(?:p|div|br|li|h[1-6]|tr|section|article)\b[^>]*>", re.I)
_TAG = re.compile(r"<[^>]+>")
_SCRIPT = re.compile(r"<(script|style|noscript)\b.*?</\1>", re.S | re.I)
_TABLE = re.compile(r"<table\b.*?</table>", re.S | re.I)
_ROW = re.compile(r"<tr\b.*?</tr>", re.S | re.I)
_CELL = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.S | re.I)
_CHROME = (
    "跳到主要內容區塊",
    "法規功能按鈕區",
    "法規內容 條文檢索",
    "您的瀏覽器不支援",
    "版權所有",
    "列印時間",
    "瀏覽",
    "Download",
)


def _clean(s: str) -> str:
    return re.sub(r"[ \t　]+", " ", htmllib.unescape(_TAG.sub("", s))).strip()


def _table_rows(table_html: str) -> list[str]:
    rows = []
    for r in _ROW.findall(table_html):
        cells = [_clean(c) for c in _CELL.findall(r)]
        if any(cells):
            rows.append(cells)
    if len(rows) < 2:
        return [" ".join(c for c in rows[0] if c)] if rows else []
    header: list[str] = []
    for h in rows[0]:
        header.append(h or (header[-1] if header else ""))
    out = []
    for r in rows[1:]:
        pairs = [f"{h}：{v}" for h, v in zip(header, r, strict=False) if v]
        if pairs:
            out.append("｜".join(pairs))
    return out


def _body(page: str) -> str:
    """The law's own block: the print page's `law-reg-content`, the history page's
    `law-reg law-content`; either way cut at the page footer, which follows the last article."""
    m = re.search(
        r'<div[^>]+class="[^"]*law-reg(?:-content|\s+law-content)[^"]*"[^>]*>(.*)',
        page,
        re.S | re.I,
    )
    body = m.group(1) if m else page
    if not m:
        m = re.search(r"<body\b[^>]*>(.*)</body>", page, re.S | re.I)
        body = m.group(1) if m else page
    return re.split(r"<footer\b", body, maxsplit=1, flags=re.I)[0]


def html_text(page: str) -> str:
    page = _SCRIPT.sub(" ", page)
    body = _body(page)
    body = _TABLE.sub(lambda m: "\n" + "\n".join(_table_rows(m.group(0))) + "\n", body)
    body = _BLOCK.sub("\n", body)
    lines = []
    for ln in htmllib.unescape(_TAG.sub("", body)).splitlines():
        ln = re.sub(r"[ \t　]+", " ", ln).strip()
        if not ln or any(k in ln for k in _CHROME):
            continue
        lines.append(ln)
    return "\n".join(lines)


# -- this experiment ----------------------------------------------------------------------------


@dataclass
class LawArticle:
    num: str
    chapter: str
    section: str
    body: str


def law_text(page: str) -> str:
    return _SOURCE.sub("", _ATTACHED.sub("", html_text(page))).rstrip("\n")


def page_meta(page: str) -> tuple[str, str]:
    name, amended = _NAME.search(page), _AMENDED.search(page)
    return (
        htmllib.unescape(name.group(1)).strip() if name else "",
        amended.group(1).strip() if amended else "",
    )


def articles_on_page(page: str) -> int:
    """Article heads counted on the raw page, independent of html_text."""
    return len(set(re.findall(r">\s*(?:本條文有附件\s*)?第 (\d+(?:-\d+)?) 條\s*<", page)))


def parse_law(text: str) -> list[LawArticle]:
    """Articles with their 章 and 節; where a law has 編 (章 numbers restart in each), the 章
    is written 「第二編 消防設計 第一章 通則」. 款 headings (「第 一 款 一般規定」, the old
    version's page) are dropped from bodies."""
    heads = list(_HEAD.finditer(text))
    marks = sorted(
        [(m.start(), "part", f"第{m.group(1)}編 {m.group(2)}") for m in _PART.finditer(text)]
        + [(m.start(), "chapter", f"第{m.group(1)}章 {m.group(2)}") for m in _CHAP.finditer(text)]
        + [(m.start(), "section", f"第{m.group(1)}節 {m.group(2)}") for m in _SECT.finditer(text)]
    )
    out: list[LawArticle] = []
    part = chapter = section = ""
    mi = 0
    for i, h in enumerate(heads):
        while mi < len(marks) and marks[mi][0] < h.start():
            kind, name = marks[mi][1], marks[mi][2]
            if kind == "part":
                part, chapter, section = name, "", ""
            elif kind == "chapter":
                chapter, section = f"{part} {name}" if part else name, ""
            else:
                section = name
            mi += 1
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        body = "\n".join(
            ln
            for ln in text[h.end() : end].splitlines()
            if ln.strip()
            and not (_PART.match(ln) or _CHAP.match(ln) or _SECT.match(ln) or _SUB.match(ln))
        ).strip()
        out.append(LawArticle(h.group(1), chapter, section, body))
    return out


def law_units(prefix: str, articles: list[LawArticle], *, max_len: int = 1000) -> list[Chunk]:
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
            out.append(Chunk(header + p, [f"{prefix}:{a.num}"], meta={"version": prefix}))
    return out
