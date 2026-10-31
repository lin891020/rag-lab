"""Why did a reply miss? One cause per wrong (model, condition, question), judged by program
from the stored evidence labels, the model's quote / expression / value and the question's
intermediate values (rules.py). Order matters: the first cause that applies is reported.

    uv run python experiments/19-lookup-tool/attribute.py -> results/attribution.csv,
                                                                  results/attribution_rows.jsonl
  hard_answer        abstain question answered (should have said found=false)
  retrieval_miss     a gold row is not in the evidence (labels)            SEER: before the model
                     (17 checks it before wrong_abstain: 16's order hid misses as abstentions)
  wrong_abstain      answerable question abstained although every gold row was there
  value_copy         a gold intermediate value appears in neither the expression nor the quote
                     (±1 %, also ×/÷1000 and ×/÷100; numbers the question gives do not count)
                                                                           SEER: value error
  no_expression      expression empty / unparseable although the numbers are there
  formula            the expression evaluates to something other than the gold (number) or the
                     gold required value (ladder)                          SEER: operator error
  selection          ladder: the expression gives the right requirement but the size chosen is
                     wrong (a lookup slip after the arithmetic)
  arithmetic         number, no tool: the expression evaluates to the gold but the model wrote a
                     different value                                       SEER: execution error
  unit_only          value right, unit wrong
  other              none of the above (e.g. value off with no expression to read)

19: in the lookup cells the "evidence" is the looked-up cells, so retrieval_miss reads "a gold row
was not looked up" (the step-1 detail is in posthoc.py).

18 adds a described field next to the cause, `slip` (read_slip), and results/slips.csv: when a
reply is wrong although every gold row was in the evidence, which cell did it read? The numbers
of the expression (the value when there is no expression; never the quote, which often copies the
whole row) are matched to the gold rows: the gold cell is there -> no slip; else a cell of a gold
row in another column -> wrong_column; else the gold column in another row -> wrong_row; else
other_number (other cells match exactly, no scales). Numbers the question gives do not count.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"
QUESTIONS = HERE / "questions.jsonl"
sys.path.insert(0, str(HERE))
from calc import BadExpression, evaluate  # noqa: E402
from rules import TABLES, row_key  # noqa: E402

logger = logging.getLogger(__name__)

TOL = 0.01
SCALES = (1.0, 1000.0, 0.001, 100.0, 0.01)
_NUM = re.compile(r"\d[\d,]*\.?\d*")
CAUSES = [
    "hard_answer",
    "wrong_abstain",
    "closed_book",
    "retrieval_miss",
    "value_copy",
    "no_expression",
    "formula",
    "selection",
    "arithmetic",
    "unit_only",
    "other",
]


def numbers_in(text: str) -> list[float]:
    s = unicodedata.normalize("NFKC", text or "")
    out = []
    for m in _NUM.finditer(s):
        try:
            out.append(float(m.group(0).replace(",", "").rstrip(".")))
        except ValueError:
            continue
    return out


def _close(x: float, v: float) -> bool:
    return v != 0 and abs(x - v) <= TOL * abs(v)


def value_present(v: float, found: list[float]) -> bool:
    return any(_close(x, v * s) for s in SCALES for x in found)


TABLE_VALUE_KEYS = {"flc", "ampacity", "table_value", "temp_factor", "count_factor"}  # + pct*


def required_inputs(q: dict) -> list[float]:
    """The table values the reply should show (not the rule's multiplier, not derived numbers
    such as the required current, not numbers the question itself gives)."""
    given = [float(g) for g in q["given"].values()]
    keys = set(TABLE_VALUE_KEYS)
    if q["expect"]["kind"] == "ladder":
        keys.discard(
            "ampacity"
        )  # the model quotes the row it chose, not the gold row (that is "selection")
    out = []
    for k, v in q["inputs"].items():
        if not (k in keys or k.startswith("pct")) or not isinstance(v, (int, float)):
            continue
        if any(_close(float(v), g) for g in given):
            continue
        out.append(float(v))
    return out


def cause_of(r: dict, q: dict) -> str | None:
    if r["right"]:
        return None
    kind = q["expect"]["kind"]
    if kind == "abstain":
        return "hard_answer"
    if r.get("rows_total", 0) and r.get("rows_hit", 0) < r["rows_total"]:
        return "retrieval_miss"  # 17: checked before abstaining (16 judged wrong_abstain first)
    if not r["answered"]:
        return "wrong_abstain"
    seen = numbers_in(r.get("expression", "")) + numbers_in(r.get("quote", ""))
    if any(not value_present(v, seen) for v in required_inputs(q)):
        return "value_copy"
    try:
        ev = evaluate(r.get("expression", ""))
    except BadExpression:
        return "no_expression" if r["grade"] != "match" else "unit_only"
    gold = q["expect"]["value"]
    if kind == "number":
        if not _close(ev, gold):
            return "formula"
        if r["grade"] == "match":
            return "unit_only"
        return "arithmetic"
    target = q["inputs"].get("required", q["inputs"].get("table_value"))
    if target is not None and not _close(ev, float(target)) and not _close(ev, gold):
        return "formula"
    if r["grade"] == "match":
        return "unit_only"
    return "selection"


def _num(cell) -> float | None:
    try:
        return float(unicodedata.normalize("NFKC", str(cell)).replace(",", ""))
    except ValueError:
        return None


def read_slip(r: dict, q: dict) -> str | None:
    """Which cell a wrong reply read (see the module doc); None when it does not apply."""
    if r["right"] or not r["answered"] or q["expect"]["kind"] == "abstain":
        return None
    if not r.get("rows_total") or r.get("rows_hit", 0) < r["rows_total"]:
        return None
    given = [float(g) for g in q["given"].values() if isinstance(g, (int, float))]
    seen = numbers_in(r.get("expression", "")) or [float(r.get("value") or 0)]
    seen = [x for x in seen if not any(_close(x, g) for g in given)]
    gold = required_inputs(q)
    if q["expect"]["kind"] == "number" and "ampacity" in q["inputs"]:
        gold = list(dict.fromkeys([*gold, float(q["inputs"]["ampacity"])]))
    cells = []  # (gold row, column) of each gold value found in a gold row
    for tid, key in q["gold"]["rows"]:
        row = next((x for x in TABLES[tid]["rows"] if row_key(tid, x) == key), None)
        if row is None:
            continue
        for j, cell in enumerate(row):
            v = _num(cell)
            if v is not None and any(_close(v, g) for g in gold):
                cells.append((tid, row, j))
    if not cells:
        return None
    if all(value_present(g, seen) for g in gold):
        return None
    # other cells match exactly (no ×100 / ×1000 scales: a multiplier 0.45 is not the cell 45),
    # and only numbers that are not a gold value
    seen = [x for x in seen if not any(_close(x, g) for g in gold)]

    def hit(cands: list[float | None]) -> bool:
        return any(o is not None and _close(x, o) for o in cands for x in seen)

    for _, row, j in cells:
        if hit([_num(c) for k, c in enumerate(row) if k != j]):
            return "wrong_column"
    for tid, row, j in cells:
        if hit([_num(x[j]) for x in TABLES[tid]["rows"] if x is not row]):
            return "wrong_row"
    return "other_number"


def main() -> int:
    from rag_lab.log import setup_logging

    setup_logging("rag_lab")
    questions = {
        json.loads(ln)["id"]: json.loads(ln)
        for ln in QUESTIONS.read_text("utf-8").splitlines()
        if ln.strip()
    }
    scored = [
        json.loads(ln)
        for ln in (RESULTS / "read/scored.jsonl").read_text("utf-8").splitlines()
        if ln.strip()
    ]
    rows = []
    counts: Counter[tuple[str, str, str]] = Counter()
    slips: Counter[tuple[str, str, str]] = Counter()
    for r in scored:
        s = read_slip(r, questions[r["qid"]])
        if s is not None:
            slips[(r["model"], r["condition"], s)] += 1
        c = cause_of(r, questions[r["qid"]])
        if c is None:
            continue
        counts[(r["model"], r["condition"], c)] += 1
        rows.append(
            {
                "model": r["model"],
                "condition": r["condition"],
                "qid": r["qid"],
                "chain": r["chain"],
                "cause": c,
                "slip": s,
                "grade": r["grade"],
            }
        )
    out = [
        {"model": m, "condition": c, "cause": cause, "n": n}
        for (m, c, cause), n in sorted(counts.items())
    ]
    with (RESULTS / "attribution.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    with (RESULTS / "slips.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "condition", "slip", "n"])
        w.writerows([*k, n] for k, n in sorted(slips.items()))
    (RESULTS / "attribution_rows.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8"
    )
    logger.info("attributed", extra={"ctx": {"misses": len(rows)}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
