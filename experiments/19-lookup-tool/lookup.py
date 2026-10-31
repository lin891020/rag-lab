"""The lookup tool: the model names a (table, row, column), the program returns that cell of the
clean tables (tables_clean.json, titles with their subtitles). Nothing else: no rule of rules.py
is called, so choosing the table (60/75/90 ℃, metal/PVC), the row, the column and any tiered
arithmetic stays with the model.

Matching, in order, never a guess (a miss is reported back as such):
  table   the id after folding (～/~/-, 〇/0 -> ○), else the longest id the text contains
  row     the key after folding; else the same words and the same numbers ("2.0mm 單線" =
          "單線 2.0"); else one bare number (or "N以下") in the text that a range row holds ("700 A" -> 800以下,
          "住宅 70000" -> 住宅 3,001至120,000): the first such row in table order, the row's leading
          words must equal the text's (or the row has none). Ranges are read by rules._in_range
          (shared with the gold: tests pin every range table by hand), plus "超過 a - b".
  column  the full or short name (18's short_names) after folding; else a bare number that a
          range column holds ("8 條" -> 7-9); else the one name containing the text, or the one
          short name starting with it.
"""

from __future__ import annotations

import logging
import re
import unicodedata

from evidence import short_names, tables_view
from rules import _in_range, row_key

logger = logging.getLogger(__name__)

TABLES = tables_view(sub=True)
TABLE_NAMES = {tid: t["names"] for tid, t in TABLES.items()}
UNITS = ["mm²", "mm2", "kva", "kw", "mm", "平方", "a", "w", "條", "具", "戶", "hp", "℃", "°c"]
ABOVE_WORDS = ["以上", "超過", "至", "-", "~"]  # only a bare number (or 以下) takes the range path
_NUM = re.compile(r"\d+(?:\.\d+)?")


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = s.replace("〜", "~").replace("‐", "-").replace("–", "-").replace("—", "-")
    return s.replace(" ", "").replace(",", "").replace("註", "")


def _fold_table(s: str) -> str:
    s = _fold(s).replace("-", "~")
    return s.replace("〇", "○").replace("0", "○").replace("零", "○")


def _words_and_numbers(s: str) -> tuple[str, list[float]]:
    f = _fold(s)
    for u in UNITS:
        f = f.replace(u, "")
    return _NUM.sub("", f), [float(x) for x in _NUM.findall(f)]


def _holds(rng: str, x: float) -> bool:
    k = _fold(rng)
    m = re.fullmatch(r"超過([\d.]+)-([\d.]+)", k)
    if m:
        return float(m.group(1)) < x <= float(m.group(2))
    try:
        return _in_range(rng, x)
    # swallow: fallback -- a key that is not a range (a name) simply does not hold the number
    except ValueError:
        return False


def _split_key(key: str) -> tuple[str, str]:
    """(leading words, range part): '住宅 3,001至120,000' -> ('住宅', '3,001至120,000')."""
    m = re.search(r"超過|\d", key)
    return (key, "") if m is None else (key[: m.start()], key[m.start() :])


def _table(text: str) -> str | None:
    f = _fold_table(text)
    ids = sorted(TABLES, key=len, reverse=True)
    for tid in ids:
        if _fold_table(tid) == f:
            return tid
    return next((tid for tid in ids if _fold_table(tid) in f), None)


def _row(tid: str, text: str) -> list[str] | None:
    rows = TABLES[tid]["rows"]
    keys = [row_key(tid, r) for r in rows]
    f = _fold(text)
    for r, k in zip(rows, keys):
        if _fold(k) == f:
            return r
    wn = _words_and_numbers(text)
    for r, k in zip(rows, keys):
        if _words_and_numbers(k) == wn:
            return r
    words, nums = wn
    if len(nums) != 1 or any(w in words for w in ABOVE_WORDS):
        return None  # "超過 10" written loosely is not read as the band below (review 2026-11-01)
    words = words.replace("以下", "")
    for r, k in zip(rows, keys):
        lead, rng = _split_key(k)
        lead = _words_and_numbers(lead)[0]
        if rng and (lead == words or lead == "") and _holds(rng, nums[0]):
            return r
    return None


def _column(tid: str, text: str) -> int | None:
    names = TABLE_NAMES[tid]
    short = short_names(names)
    f = _fold(text)
    for i, (n, s) in enumerate(zip(names, short)):
        if f in (_fold(n), _fold(s)):
            return i
    words, nums = _words_and_numbers(text)
    if not words and len(nums) == 1:
        for i, s in enumerate(short):
            if _holds(s, nums[0]):
                return i
    if not f:
        return None
    having = [i for i, n in enumerate(names) if f in _fold(n)]
    if len(having) == 1:
        return having[0]
    starting = [i for i, s in enumerate(short) if _fold(s).startswith(f)]
    return starting[0] if len(starting) == 1 else None


def resolve(table: str, row: str, column: str) -> dict:
    """One lookup -> {asked, table, row, column, status, value, label}; status is ok, blank (the
    cell is empty), no_table, no_row or no_column. A dash is a value ("-"), not a blank."""
    out = {
        "asked": [table, row, column],
        "table": None,
        "row": None,
        "column": None,
        "status": "",
        "value": "",
        "label": "",
    }
    tid = _table(table)
    if tid is None:
        return {**out, "status": "no_table"}
    r = _row(tid, row)
    if r is None:
        return {**out, "table": tid, "status": "no_row"}
    key = row_key(tid, r)
    c = _column(tid, column)
    if c is None:
        return {**out, "table": tid, "row": key, "label": f"{tid}|{key}", "status": "no_column"}
    v = str(r[c]).strip() if c < len(r) else ""
    return {
        **out,
        "table": tid,
        "row": key,
        "column": TABLE_NAMES[tid][c],
        "label": f"{tid}|{key}",
        "status": "ok" if v else "blank",
        "value": v,
    }


def render(results: list[dict]) -> str:
    """The cells for the second step, grouped by table in the order first asked: the table's title
    line, one line per lookup (`表｜列｜欄 = 值`, blanks and misses said in words), its notes."""
    order: list[str | None] = []
    for r in results:
        if r["table"] not in order:
            order.append(r["table"])
    lines = []
    for tid in order:
        if tid is None:
            lines += [f"查無此表：{r['asked'][0]}" for r in results if r["table"] is None]
            continue
        t = TABLES[tid]
        lines.append(f"{tid} {t['title'].removeprefix(tid).strip()}")
        for r in results:
            if r["table"] != tid:
                continue
            if r["status"] == "no_row":
                lines.append(f"{tid} 查無此列：{r['asked'][1]}")
            elif r["status"] == "no_column":
                lines.append(f"{tid}｜{r['row']} 查無此欄：{r['asked'][2]}")
            else:
                v = r["value"] if r["status"] == "ok" else "（空白）"
                lines.append(f"{tid}｜{r['row']}｜{r['column']} = {v}")
        if t.get("notes"):
            n = t["notes"].strip()
            lines.append(f"{tid} {n}" if n.startswith("註") else f"{tid} 註：{n}")
    return "\n".join(lines)


def catalog() -> str:
    """Every clean table: id and title (with subtitle), column names, row names. No values."""
    out = []
    for tid, t in TABLES.items():
        keys = [row_key(tid, r) for r in t["rows"]]
        out.append(
            f"{tid} {t['title'].removeprefix(tid).strip()}\n"
            f"  欄：{'、'.join(t['names'])}\n"
            f"  列：{'、'.join(keys)}"
        )
    return "\n".join(out)
