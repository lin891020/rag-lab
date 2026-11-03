"""Which table cells a C1/C4/C5 question needs, and whether a step-2 evidence text shows them.
Shared by run_read (21's hit fields) and posthoc; works on fix and seg text alike, since both
write one `表｜列｜欄 = 值` line per cell (seg marks the asked rows with ← 你查的)."""

from __future__ import annotations

import re

TOL = 0.01
_LINE = re.compile(r"^(?P<t>[^｜\s]+)｜(?P<k>[^｜]+)｜(?P<c>[^｜]+) = (?P<v>.*?)(?: ← 你查的)?$")


def num(s: str) -> float | None:
    try:
        return float(str(s).replace(",", ""))
    except ValueError:
        return None


def near(x: float, g: float) -> bool:
    return abs(x - g) <= TOL * max(1.0, abs(g))


def gold_cells(q: dict) -> list[float]:
    """The table cells an answerable C1/C4/C5 question needs, from its intermediate values."""
    inp = q["inputs"]
    if q["chain"] == "C1":
        return [inp["ampacity"]]
    if q["chain"] == "C5":
        return [inp["table_value"]]
    unused = {"pct_rest"} if inp.get("rest") == 0 else {"pct2"} if inp.get("band2") == 0 else set()
    cells = [round(v * 100, 6) for k, v in inp.items() if k.startswith("pct") and k not in unused]
    return cells + ([100.0] if "first" in inp else [])


def gold_labels(q: dict) -> list[str]:
    """The gold rows as `表|列`, without the band a question never reaches (c4-16)."""
    gold = [f"{t}|{k}" for t, k in q["gold"]["rows"]]
    if q["inputs"].get("rest") == 0 or q["inputs"].get("band2") == 0:
        gold = gold[:1]
    return gold


def shown_cells(text: str) -> list[tuple[str, str, float]]:
    """(`表|列`, column, value) for every numeric cell line of an evidence text."""
    out = []
    for ln in text.splitlines():
        m = _LINE.match(ln)
        if m and (v := num(m["v"])) is not None:
            out.append((f"{m['t']}|{m['k']}", m["c"], v))
    return out


def shown(text: str, q: dict) -> bool:
    """Every gold row and every gold cell value is in the text (any column of a gold row)."""
    cells = shown_cells(text)
    gold = gold_labels(q)
    labels = {lb for lb, _, _ in cells}
    if any(lb not in labels for lb in gold):
        return False
    got = [v for lb, _, v in cells if lb in gold]
    return all(any(near(v, g) for v in got) for g in gold_cells(q))
