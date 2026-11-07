"""The 114-05-09 修正條文對照表 (PDF) -> corpus/rows.json, one row per table row.

    uv run python experiments/29-amendment-diff/compare_table.py

The PDF draws the table: three columns (修正條文 / 現行條文 / 說明) between vertical rules at
x = 80, 230, 380, 530 pt, rows between horizontal rules (0.5 pt rects, ~150 pt wide, the same
tops in all three columns). A row is the band between two rules on one page; the first band of a
page that starts no article (no 「第N條」 and no 「第N章／節」 word in the two text columns) goes
on from the last band of the page before. Each column's text is the crop of the band.
A header word is 「第N條」 standing alone as the first word of its cell (pdfplumber splits words
at spaces, and the PDF puts a space after the article number); a wrapped 「第二十五條規定」 is one
word and not a header, and a reference that happens to end a line (「依第一百五十二條」) is not the
cell's first word. Lower in a cell, a header counts only at the cell's left edge with the line
above ending a sentence (merged articles: 新 370 ← 舊 97 + 96 sit in one old cell). One old
article may also start several rows (split over new articles), and a cell may map only part of
an article (「第一百七十一條第一款、…」, flagged old_part; a long header wrapped onto the next line,
「第二百五十二條之三第二」／「項」, is read across the two lines). An added article has an empty old
cell; a deleted one an empty new cell. The table's rules follow whichever column starts
articles, and the other column's text may run on across a rule: a band whose old column starts
an article while the new text runs on joins the row above unless the 說明 starts afresh (a merged
article, 新 281 ← 舊 185-5 + 396-74); a band that starts a new article while the old column's
text runs on before its own header gives that leading text to the row above (新 871／872).
Numbers are normalised (articles.norm_article): 第十三條之一 -> 13-1.

Pages 917–1,081 (landscape) compare the 附表／附圖; they have no row rules and are not read here
(the plan asks only about articles). The portrait pages with rules are the article table.

rows.json: [{kind: article|structure|other, new_nums, old_nums, new_text, old_text, note,
pages, marks}], plus a trailing meta object with counts.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pdfplumber

from rag_lab.log import setup_logging

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from articles import norm_article  # noqa: E402
import laws  # noqa: E402

logger = logging.getLogger(__name__)

ROOT = HERE.parents[1]
COMPARE_PDF = ROOT / "data/29-amendment-diff/compare-1140509.pdf"
OLD_TXT = HERE / "corpus/old-1100317.txt"
ROWS = HERE / "corpus/rows.json"
COLS = {"new": (80.0, 230.0), "old": (230.0, 380.0), "note": (380.0, 530.0)}  # pt, the rules
RULE_WIDTH = (140.0, 160.0)  # a row rule is a 0.5 pt rect about one column wide
_CN = "零〇一二三四五六七八九十百千"
_HEAD = re.compile(
    rf"^第([{_CN}]+條(?:之[{_CN}]+)?)(第[{_CN}]+[項款目].*|序文.*|前段.*|後段.*|但書.*)?$"
)
_DANGLING = re.compile(
    rf"^第([{_CN}]+條(?:之[{_CN}]+)?)第[{_CN}]*$"
)  # header wrapped: 「…之三第二」／「項」
_PART_START = re.compile(rf"^(第[{_CN}]+[項款目]|[{_CN}]*[項款目]|序文|前段|後段|但書)")
_MARK = re.compile(rf"^第[{_CN}]+[章節款]$")
COLUMN_HEADER = ("修正條文", "現行條文", "說明")
MARGIN = 8.0  # pt: a header word starts at the cell's left edge (body lines may be indented)
SENTENCE_END = "。）)：；」"  # the line before a second header in one cell ends a sentence


def head_num(word: str) -> str | None:
    """「第六條」 -> "6", 「第十三條之一」 -> "13-1", 「第一百七十一條第一款、」 -> "171" (part of an
    article); anything else None."""
    m = _HEAD.match(word)
    return norm_article("第" + m.group(1)) if m else None


def head_part(word: str) -> bool:
    """「第一百七十一條第一款、」: the row maps part of the article."""
    m = _HEAD.match(word)
    return bool(m and m.group(2))


def _head_at(lines: list[tuple[float, str, str]], i: int) -> tuple[str, bool] | None:
    """The article a line starts, if any: the header word alone, or wrapped onto the next line
    (「第二百五十二條之三第二」／「項」); part when a 項／款／目／序文 follows on the line or the next."""
    _x0, word, text = lines[i]
    nxt = lines[i + 1][2].replace(" ", "") if i + 1 < len(lines) else ""
    num = head_num(word)
    if num:
        rest = text.replace(" ", "")[len(word) :]
        part = (
            head_part(word)
            or bool(_PART_START.match(rest))
            or (not rest and bool(_PART_START.match(nxt)))
        )
        return num, part
    m = _DANGLING.match(word)
    if m and nxt and re.match(rf"^[{_CN}]*[項款目]", nxt):
        return norm_article("第" + m.group(1)), True
    return None


def mark_word(word: str) -> bool:
    return bool(_MARK.match(word))


def cell_heads(lines: list[tuple[float, str, str]], left: float) -> list[tuple[str, bool, int]]:
    """Article numbers a cell starts, from its lines as (x0 of the first word, first word, line
    text): (number, part, line index). The first line's first word; lower down only a word at the
    left edge whose line above ends a sentence (a second article in a merged cell), never a
    reference that ends a line."""
    out = []
    for i, (x0, _word, _text) in enumerate(lines):
        if x0 > left + MARGIN:
            continue
        head = _head_at(lines, i)
        if head is None:
            continue
        if i == 0 or lines[i - 1][2].rstrip()[-1:] in SENTENCE_END:
            out.append((*head, i))
    return out


_DROP = re.compile(r"[\s\W_]+")  # whitespace and punctuation: the PDF writes 二‧五, the page 二．五
_HEAD_AT_START = re.compile(rf"^第[{_CN}]+條(之[{_CN}]+)?")


def fold_text(s: str) -> str:
    """Letters, digits and CJK only, NFKC-folded: what two spellings of one article share."""
    return _DROP.sub("", unicodedata.normalize("NFKC", s))


def old_body(row: dict) -> str:
    """The old cell without its 「第N條」 header, folded."""
    return _HEAD_AT_START.sub("", fold_text(row["old_text"]), count=1)


def align(rows: list[dict], old_articles: dict[str, str]) -> dict:
    """Rows mapping exactly one whole old article: does the cell's text equal the old version's
    article (folded)? Sets row["old_aligned"]; returns the counts. The plan's gate is >= 0.90."""
    n = same = 0
    for r in rows:
        if r["kind"] != "article" or len(r["old_nums"]) != 1 or r.get("old_part"):
            continue
        body = old_articles.get(r["old_nums"][0])
        if body is None:
            continue
        n += 1
        r["old_aligned"] = old_body(r) == fold_text(body)
        same += r["old_aligned"]
    return {"compared": n, "aligned": same, "rate": round(same / n, 4) if n else 0.0}


def sort_key(num: str) -> tuple[int, int]:
    a, _, b = num.partition("-")
    return int(a), int(b or 0)


def _col_of(x0: float, x1: float) -> str | None:
    for name, (a, b) in COLS.items():
        if a - 1 <= x0 and x1 <= b + 1:
            return name
    return None


def crossing_words(page) -> list[str]:
    """Words inside the table that do not sit in one column (the plan's check: must be empty)."""
    out = []
    for w in page.extract_words():
        if w["x0"] < COLS["new"][0] - 1 or w["x1"] > COLS["note"][1] + 1:
            continue  # outside the table: title, page number
        if _col_of(w["x0"], w["x1"]) is None:
            out.append(w["text"])
    return out


def row_rules(page) -> list[float]:
    tops = sorted(
        r["top"]
        for r in page.rects
        if r["height"] < 1 and RULE_WIDTH[0] < r["width"] < RULE_WIDTH[1]
    )
    out: list[float] = []
    for t in tops:
        if not out or t - out[-1] > 1.0:
            out.append(t)
    return out


def page_bands(page, n: int) -> list[dict]:
    """The row bands of one page: cell texts, article numbers and 章／節 marks per column."""
    rules = row_rules(page)
    if len(rules) < 2:
        return []
    words = page.extract_words()
    out = []
    for a, b in zip(rules, rules[1:]):
        cells, heads, marks = {}, {"new": [], "old": []}, []
        for col, (x0, x1) in COLS.items():
            cells[col] = (page.crop((x0, a, x1, b)).extract_text() or "").strip()
        by_line: dict[str, dict[float, list]] = {"new": {}, "old": {}}
        for w in words:
            if not (a <= w["top"] < b):
                continue
            col = _col_of(w["x0"], w["x1"])
            if col in ("new", "old"):
                by_line[col].setdefault(round(w["top"]), []).append(w)
        parts: dict[str, list[bool]] = {"new": [], "old": []}
        first = {"new": False, "old": False}  # does the cell's first line start an article/章節?
        split: dict[str, tuple[str, str]] = {}  # text above the first header / from it on
        for col in ("new", "old"):
            tops = sorted(by_line[col])
            lines = []
            for top in tops:
                ws = sorted(by_line[col][top], key=lambda w: w["x0"])
                lines.append((ws[0]["x0"], ws[0]["text"], " ".join(w["text"] for w in ws)))
            found = cell_heads(lines, COLS[col][0])
            for num, part, _i in found:
                heads[col].append(num)
                parts[col].append(part)
            if lines:
                first[col] = _head_at(lines, 0) is not None or mark_word(lines[0][1])
                if mark_word(lines[0][1]) and lines[0][1] not in marks:
                    marks.append(lines[0][1])
            if found and not first[col] and found[0][2] > 0:  # the column runs on from the row
                # above, then starts an article: the top part belongs to the row above
                cut_at = tops[found[0][2]] - 1.0
                x0, x1 = COLS[col]
                split[col] = (
                    (page.crop((x0, a, x1, cut_at)).extract_text() or "").strip(),
                    (page.crop((x0, cut_at, x1, b)).extract_text() or "").strip(),
                )
        out.append(
            {
                "page": n,
                "cells": cells,
                "heads": heads,
                "parts": parts,
                "marks": marks,
                "first": first,
                "split": split,
            }
        )
    return out


def read_pages(pdf_path: Path) -> list[list[dict]]:
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            if page.width > page.height:
                continue  # the 附表／附圖 pages: no row rules, not read
            bands = page_bands(page, i)
            if bands:
                pages.append(bands)
            if i % 100 == 0:
                logger.info("pages read", extra={"ctx": {"page": i}})
    return pages


_FRESH_NOTE = re.compile(
    rf"^\s*(一\s*、|本條|條次|本節|本章|本款|節名|章名|款名|第[{_CN}]+[章節款])"
)


def _fresh_note(note: str) -> bool:
    """Does the 說明 cell start a new entry (「一、…」「本條…」「條次變更…」) rather than run on
    mid-sentence from the row above?"""
    first = note.strip().splitlines()[0] if note.strip() else ""
    return bool(_FRESH_NOTE.match(first))


def _starts(band: dict) -> bool:
    """Does a band start a row? Yes when the new column or a 章／節 mark starts it, or the old
    column starts an article beside an empty new cell (a deleted article). When the old column
    starts an article while the new text runs on from the row above, the table has ruled inside
    a merged article (新 281 ← 舊 185-5 + 396-74) unless the 說明 starts afresh."""
    first = band.get("first")
    if first is None:  # synthetic bands without the flag
        return bool(band["heads"]["new"] or band["heads"]["old"] or band["marks"])
    c = band["cells"]
    if first["new"] or band["marks"]:
        return True
    if first["old"]:
        return not c["new"].strip() or _fresh_note(c["note"])
    return False


def _join(a: str, b: str) -> str:
    return f"{a}\n{b}".strip("\n") if a and b else a or b


def _append(r: dict, band: dict, new: str, old: str, note: str) -> None:
    r["new_text"] = _join(r["new_text"], new)
    r["old_text"] = _join(r["old_text"], old)
    r["note"] = _join(r["note"], note)
    if band["page"] not in r["pages"]:
        r["pages"].append(band["page"])


def assemble(pages: list[list[dict]]) -> list[dict]:
    rows: list[dict] = []
    for bands in pages:
        for band in bands:
            c = band["cells"]
            if tuple(c[k] for k in ("new", "old", "note")) == COLUMN_HEADER:
                continue
            if rows and not _starts(band):  # the row above goes on (across a page or a rule)
                r = rows[-1]
                _append(r, band, c["new"], c["old"], c["note"])
                r["new_nums"] += band["heads"]["new"]  # a second article lower in the cell
                r["old_nums"] += band["heads"]["old"]
                r["old_part"] = r["old_part"] or any(band.get("parts", {}).get("old", []))
                continue
            split = band.get("split", {})
            cells = dict(c)
            if rows and split:  # a column ran on from the row above before its own article starts
                above = {col: split[col][0] for col in split}
                _append(rows[-1], band, above.get("new", ""), above.get("old", ""), "")
                for col in split:
                    cells[col] = split[col][1]
            heads = band["heads"]
            kind = (
                "article"
                if heads["new"] or heads["old"]
                else "structure"
                if band["marks"]
                else "other"
            )
            rows.append(
                {
                    "kind": kind,
                    "new_nums": list(heads["new"]),
                    "old_nums": list(heads["old"]),
                    "old_part": any(band.get("parts", {}).get("old", [])),
                    "new_text": cells["new"],
                    "old_text": cells["old"],
                    "note": cells["note"],
                    "pages": [band["page"]],
                    "marks": list(band["marks"]),
                }
            )
    return rows


def stats(rows: list[dict]) -> dict:
    art = [r for r in rows if r["kind"] == "article"]
    return {
        "rows": len(rows),
        "article_rows": len(art),
        "structure_rows": sum(r["kind"] == "structure" for r in rows),
        "other_rows": sum(r["kind"] == "other" for r in rows),
        "new_nums": sum(len(r["new_nums"]) for r in art),
        "old_nums": sum(len(r["old_nums"]) for r in art),
        "one_to_one": sum(len(r["new_nums"]) == 1 and len(r["old_nums"]) == 1 for r in art),
        "added": sum(bool(r["new_nums"]) and not r["old_nums"] for r in art),
        "deleted": sum(bool(r["old_nums"]) and not r["new_nums"] for r in art),
        "old_in_several_rows": sum(
            v > 1 for v in Counter(n for r in art for n in r["old_nums"]).values()
        ),
        "many_old_in_one_row": sum(len(r["old_nums"]) > 1 for r in art),
        "old_part_rows": sum(bool(r.get("old_part")) for r in art),
        "pages_spanned": Counter(len(r["pages"]) for r in art).most_common(),
    }


def main() -> int:
    setup_logging("rag_lab")
    rows = assemble(read_pages(COMPARE_PDF))
    old = {a.num: a.body for a in laws.parse_law(OLD_TXT.read_text("utf-8"))}
    s = {**stats(rows), "alignment": align(rows, old)}
    logger.info("rows", extra={"ctx": s})
    ROWS.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", "utf-8")
    (HERE / "corpus/rows-stats.json").write_text(
        json.dumps(s, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
