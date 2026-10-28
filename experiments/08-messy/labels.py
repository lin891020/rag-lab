"""Chunk labels that survive the mess.

01's labels come from parse_articles() on the extracted text. A header, a footer or a
two-column interleave can break the `第 N 條` lines and the parse silently loses articles, which
would make a retrieval drop look like a chunking drop. So 08 labels chunks by witness lines:
the clean text is split into lines, each line is assigned to the article running at that point,
and a chunk gets every article that has at least one witness line (>= MIN_LINE chars) inside it.
The two-column typesetter keeps the same (line, label) pairs after wrapping.
"""

from __future__ import annotations

import bisect
import logging
import re

from chunkers import Chunk

logger = logging.getLogger(__name__)

MIN_LINE = 8
_HEAD = re.compile(r"^第 (\d+) 條$")


def line_map_from_text(text: str) -> list[tuple[str, str]]:
    """(line, article number or '') for every line of the clean text, in order."""
    out: list[tuple[str, str]] = []
    label = ""
    for ln in text.splitlines():
        m = _HEAD.match(ln)
        if m:
            label = m.group(1)
        out.append((ln, label))
    logger.debug("line map", extra={"ctx": {"lines": len(out)}})
    return out


def witnesses(line_map: list[tuple[str, str]]) -> dict[str, str]:
    """line -> article, keeping only content lines of >= MIN_LINE chars that belong to exactly
    one article. Boilerplate such as 「依下列規定辦理：」 recurs across articles and would label
    everything; `第 N 條` header lines are excluded because a second regulation (the concat
    variant) has the same headers. On the clean text this still labels all 1050 article chunks
    exactly as the parse does (tests)."""
    owners: dict[str, set[str]] = {}
    for line, label in line_map:
        line = line.strip()
        if label and len(line) >= MIN_LINE and not _HEAD.match(line):
            owners.setdefault(line, set()).add(label)
    return {line: next(iter(labs)) for line, labs in owners.items() if len(labs) == 1}


def _boundary(s: str, i: int) -> bool:
    return i < 0 or i >= len(s) or s[i] in "\n \x1f"


def label_chunks(chunks: list[Chunk], line_map: list[tuple[str, str]]) -> list[Chunk]:
    """Return chunks with labels = sorted articles witnessed inside their text. The chunk texts
    are joined once and searched with str.find, then offsets map back to chunks by bisect."""
    sep = "\x1f"
    texts = [c.text for c in chunks]
    joined = sep.join(texts)
    starts: list[int] = []
    pos = 0
    for t in texts:
        starts.append(pos)
        pos += len(t) + 1
    found: list[set[str]] = [set() for _ in chunks]
    for line, label in witnesses(line_map).items():
        i = joined.find(line)
        while i != -1:
            # whole line only: 「依下列規定辦理：」 is also the tail of hundreds of longer lines.
            # A space counts as a boundary because pdfplumber glues two columns with one.
            if _boundary(joined, i - 1) and _boundary(joined, i + len(line)):
                found[bisect.bisect_right(starts, i) - 1].add(label)
            i = joined.find(line, i + 1)
    out = [
        Chunk(c.text, sorted(f, key=int), c.start, c.end, dict(c.meta))
        for c, f in zip(chunks, found, strict=True)
    ]
    logger.debug(
        "labelled",
        extra={"ctx": {"chunks": len(out), "unlabelled": sum(not c.labels for c in out)}},
    )
    return out
