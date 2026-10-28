"""Pick the data rows a question is about, from the question text alone.

Score = (number of cells whose parsed range contains a number from the question,
         char-bigram BM25 of the question against the row's kv text). Top-k rows, original order.
The gold answer is never consulted; `row` in questions.jsonl is only used afterwards to report
how often the right row was kept (row recall).
"""

from __future__ import annotations

import math
import re

from tables import Table, parse_range

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def question_numbers(q: str) -> list[float]:
    return [float(m.replace(",", "")) for m in _NUM.findall(q)]


def bigrams(s: str) -> list[str]:
    s = re.sub(r"\s+", "", s)
    return [s[i : i + 2] for i in range(len(s) - 1)] or ([s] if s else [])


def row_text(t: Table, i: int) -> str:
    names = t.column_names()
    return "；".join(f"{n}={v}" for n, v in zip(names, t.data_rows()[i], strict=True) if v)


def _bm25(
    query: list[str], docs: list[list[str]], *, k1: float = 1.5, b: float = 0.75
) -> list[float]:
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / max(1, n)
    df: dict[str, int] = {}
    for d in docs:
        for g in set(d):
            df[g] = df.get(g, 0) + 1
    out = []
    for d in docs:
        tf: dict[str, int] = {}
        for g in d:
            tf[g] = tf.get(g, 0) + 1
        s = 0.0
        for g in query:
            if g not in tf:
                continue
            idf = math.log(1 + (n - df[g] + 0.5) / (df[g] + 0.5))
            s += idf * tf[g] * (k1 + 1) / (tf[g] + k1 * (1 - b + b * len(d) / avgdl))
        out.append(s)
    return out


def _contains(cell: str, x: float) -> bool:
    r = parse_range(cell)
    if r is None:
        return False
    lo, hi = r
    return (lo is None or x >= lo) and (hi is None or x <= hi)


def select_rows(t: Table, question: str, *, k: int = 3) -> list[int]:
    rows = t.data_rows()
    if len(rows) <= k:
        return list(range(len(rows)))
    nums = question_numbers(question)
    hits = [sum(any(_contains(c, x) for x in nums) for c in r) for r in rows]
    sims = _bm25(bigrams(question), [bigrams(row_text(t, i)) for i in range(len(rows))])
    order = sorted(range(len(rows)), key=lambda i: (-hits[i], -sims[i], i))
    return sorted(order[:k])


def subtable_markdown(t: Table, idxs: list[int]) -> str:
    names = t.column_names()
    rows = t.data_rows()
    lines = [t.title, "| " + " | ".join(names) + " |", "|" + "---|" * len(names)]
    for i in idxs:
        lines.append("| " + " | ".join(rows[i]) + " |")
    if t.notes:
        lines.append(t.notes)
    return "\n".join(lines)
