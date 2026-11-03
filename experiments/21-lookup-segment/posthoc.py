"""Numbers computed after the reading track, described only (no verdicts): whether the gold cells
were in what the tool showed (shown) and how often a wrong answer used a value shown for another
row (read_slip), per cell and chain; where 19's step 1 was lost (step1_class, the same for seg);
gpt-oss asked twice. Reads results/read/{scored,raw}.jsonl; writes results/posthoc.md.

    uv run python experiments/21-lookup-segment/posthoc.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

from rag_lab.log import setup_logging

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import cells  # noqa: E402
from cells import TOL, gold_cells  # noqa: E402, F401
from cells import num as _num  # noqa: E402
from run_read import COPIED, RUNS, read_raw  # noqa: E402
from score import bootstrap_ci, judge, paired_diff_ci  # noqa: E402

logger = logging.getLogger(__name__)

SCORED = HERE / "results" / "read" / "scored.jsonl"
RAW = HERE / "results" / "read" / "raw.jsonl"
QUESTIONS = HERE / "questions.jsonl"
OUT = HERE / "results" / "posthoc.md"
CELLS = {m: COPIED[m] + RUNS[m] for m in RUNS}  # model -> its cells, fix* copied from 19
MODELS = list(CELLS)
CLASSES = ["no_lookups", "wrong_table", "wrong_row", "wrong_column", "cells_ok"]
CLASS_NAME = {
    "no_lookups": "沒查",
    "wrong_table": "選錯表",
    "wrong_row": "選錯列",
    "wrong_column": "選錯欄",
    "cells_ok": "正解格都查到",
}
STATUSES = ["ok", "blank", "no_table", "no_row", "no_column"]


def _f(x: float) -> str:
    return f"{x:+.2f}".replace("+0.", "+.").replace("-0.", "−.")


def _r(x: float) -> str:
    return f"{x:.2f}".lstrip("0") if x < 1 else "1.0"


def step1_class(row: dict, q: dict) -> str:
    """The first stage of the lookup that missed: a gold table not looked up, a gold row not
    reached, a gold cell value not among the cells returned; else cells_ok."""
    res = row.get("lookups") or []
    if not res:
        return "no_lookups"
    gold = [(t, k) for t, k in q["gold"]["rows"]]
    if q["inputs"].get("rest") == 0 or q["inputs"].get("band2") == 0:
        gold = gold[:1]  # c4-16: nothing falls in the upper band, so its row is not needed
    tables = {r["table"] for r in res}
    if any(t not in tables for t, _ in gold):
        return "wrong_table"
    # a known row with an unknown column reached the row: it fails below, as a column (19 review)
    labels = {r["label"] for r in res if r["status"] in ("ok", "blank", "no_column")}
    if any(f"{t}|{k}" not in labels for t, k in gold):
        return "wrong_row"
    got = [v for r in res if r["status"] == "ok" and (v := _num(r["value"])) is not None]
    for g in gold_cells(q):
        if not any(abs(v - g) <= TOL * max(1.0, abs(g)) for v in got):
            return "wrong_column"
    return "cells_ok"


CONSTANTS = {100.0, 1000.0}
_NUMS = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


def shown(row: dict, q: dict) -> bool:
    return cells.shown(row.get("evidence_text") or "", q)


def read_slip(row: dict, q: dict) -> bool:
    """A wrong answer, gold cells shown, whose expression misses a gold cell and uses instead a
    value shown in a gold column for a non-gold row (as is or as a percent). Values of the gold
    rows (any column), the question's own numbers and 100 / 1000 (percent, kilo) never count."""
    if row.get("right"):
        return False
    text = row.get("evidence_text") or ""
    if not cells.shown(text, q):
        return False
    shown_ = cells.shown_cells(text)
    gold = set(cells.gold_labels(q))
    want = cells.gold_cells(q)
    cols = {c for lb, c, v in shown_ if lb in gold and any(cells.near(v, g) for g in want)}
    mine = [v for lb, _, v in shown_ if lb in gold]
    mine += [float(x.replace(",", "")) for x in _NUMS.findall(q["question"])]
    other = [v for lb, c, v in shown_ if lb not in gold and c in cols]
    used = [float(s.replace(",", "")) for s in _NUMS.findall(str(row.get("expression") or ""))]
    used = [x for x in used if x not in CONSTANTS]  # ÷ 100 for a percent is not a table value
    forms = [f for x in used for f in (x, x * 100)]
    if all(any(cells.near(f, g) for f in forms) for g in want):
        return False  # every gold cell is in the expression: the slip is elsewhere
    return any(
        cells.near(f, v) and not any(cells.near(f, m) for m in mine) for f in forms for v in other
    )


def classes(rows: list[dict], questions: dict[str, dict]) -> list[str]:
    out = [
        "| 模型 | 格 | " + " | ".join(CLASS_NAME[c] for c in CLASSES) + " |",
        "|---|---|" + "---|" * len(CLASSES),
    ]
    for m in MODELS:
        for c in (c for c in CELLS[m] if c.startswith("fix")):
            x = [
                r for r in rows if (r["model"], r["condition"]) == (m, c) and r["kind"] != "abstain"
            ]
            if not x:
                continue
            by: dict[str, list[bool]] = {k: [] for k in CLASSES}
            for r in x:
                by[step1_class(r, questions[r["qid"]])].append(bool(r["right"]))
            cells = [f"{len(v)}（答對 {sum(v)}）" for v in by.values()]
            out.append(f"| {m} | {c} | " + " | ".join(cells) + " |")
    return out


def by_chain_class(rows: list[dict], questions: dict[str, dict]) -> list[str]:
    out = ["| 模型 | 鏈 | " + " | ".join(CLASS_NAME[c] for c in CLASSES) + " |"]
    out.append("|---|---|" + "---|" * len(CLASSES))
    for cell in ("fix",):
        for m in MODELS:
            for ch in ("C1", "C4", "C5"):
                x = [
                    r
                    for r in rows
                    if (r["model"], r["condition"], r["chain"]) == (m, cell, ch)
                    and r["kind"] != "abstain"
                ]
                if not x:
                    continue
                got = Counter(step1_class(r, questions[r["qid"]]) for r in x)
                out.append(
                    f"| {m} {cell} | {ch} | " + " | ".join(str(got[c]) for c in CLASSES) + " |"
                )
    return out


def statuses(raw: list[dict]) -> list[str]:
    out = [
        "| 模型 | 格 | 每題查幾格（平均） | " + " | ".join(STATUSES) + " | 第一步有問題 |",
        "|---|---|---|" + "---|" * len(STATUSES) + "---|",
    ]
    for m in MODELS:
        for c in (c for c in CELLS[m] if c.startswith("fix")):
            x = [r for r in raw if (r["model"], r["condition"]) == (m, c) and not r.get("error")]
            if not x:
                continue
            got = Counter(s["status"] for r in x for s in r["lookups"])
            probs = Counter(r["step1_problem"] for r in x if r["step1_problem"])
            n = sum(got.values()) / len(x)
            out.append(
                f"| {m} | {c} | {n:.1f} | "
                + " | ".join(str(got[s]) for s in STATUSES)
                + f" | {dict(probs) or '—'} |"
            )
    return out


def twice(raw: list[dict], questions: dict[str, dict], model: str, cell: str) -> list[str]:
    a = {r["qid"]: r for r in raw if (r["model"], r["condition"]) == (model, cell)}
    b = {r["qid"]: r for r in raw if (r["model"], r["condition"]) == (model, f"{cell}-2")}
    qids = sorted(set(a) & set(b))
    if not qids:
        return [f"| {model} {cell} | （沒有第二次） | | | | |"]

    def ans(r: dict) -> tuple:
        j = judge(r["raw_content"], questions[r["qid"]])
        return j["answered"], j["value"]

    same = sum(a[q]["raw_content"] == b[q]["raw_content"] for q in qids)
    one = "step1_content" in a[qids[0]]
    same1 = sum(a[q]["step1_content"] == b[q]["step1_content"] for q in qids) if one else None
    same_ans = sum(ans(a[q]) == ans(b[q]) for q in qids)
    ids = [q for q in qids if questions[q]["expect"]["kind"] != "abstain"]
    x = [judge(b[q]["raw_content"], questions[q])["right"] for q in ids]
    y = [judge(a[q]["raw_content"], questions[q])["right"] for q in ids]
    d, lo, hi = paired_diff_ci(x, y)
    flips = sum(p != q for p, q in zip(x, y, strict=True))
    first = f"{same1}/{len(qids)}" if one else "—"
    return [
        f"| {model} {cell} | {first} | {same}/{len(qids)} | {same_ans}/{len(qids)} "
        f"| {_f(d)} [{_f(lo)}, {_f(hi)}]（{len(ids)}） | {flips} |"
    ]


TWICE_HEAD = [
    "| 模型 格 | 第一步一字不差 | 回覆一字不差 | 答案（found, value）相同 "
    "| 答對 第二次 − 第一次 [95%] | 對錯翻面 |",
    "|---|---|---|---|---|---|",
]


def shown_table(rows: list[dict], questions: dict[str, dict]) -> list[str]:
    out = [
        "| 模型 | 格 | 正解格在回傳裡（其中答對） | 不在（其中答對） | 讀錯列 | 回傳字數（平均／最大） |",
        "|---|---|---|---|---|---|",
    ]
    for m in MODELS:
        for c in CELLS[m]:
            x = [
                r for r in rows if (r["model"], r["condition"]) == (m, c) and r["kind"] != "abstain"
            ]
            if not x:
                continue
            yes = [r for r in x if shown(r, questions[r["qid"]])]
            no = [r for r in x if not shown(r, questions[r["qid"]])]
            slips = sum(read_slip(r, questions[r["qid"]]) for r in yes)
            n = [r.get("evidence_chars", 0) for r in x]
            out.append(
                f"| {m} | {c} | {len(yes)}（{sum(r['right'] for r in yes)}） "
                f"| {len(no)}（{sum(r['right'] for r in no)}） | {slips} "
                f"| {sum(n) / len(n):.0f}／{max(n)} |"
            )
    return out


def shown_by_chain(rows: list[dict], questions: dict[str, dict]) -> list[str]:
    out = ["| 模型 格 | 鏈 | 正解格在回傳裡 | 答對 | 讀錯列 |", "|---|---|---|---|---|"]
    for m in MODELS:
        for c in ("fix", "seg"):
            for ch in ("C1", "C4", "C5"):
                x = [
                    r
                    for r in rows
                    if (r["model"], r["condition"], r["chain"]) == (m, c, ch)
                    and r["kind"] != "abstain"
                ]
                if not x:
                    continue
                yes = sum(shown(r, questions[r["qid"]]) for r in x)
                slips = sum(read_slip(r, questions[r["qid"]]) for r in x)  # 0 unless shown
                right = sum(bool(r["right"]) for r in x)
                out.append(f"| {m} {c} | {ch} | {yes}/{len(x)} | {right}/{len(x)} | {slips} |")
    return out


def right_when_ok(rows: list[dict], questions: dict[str, dict]) -> list[str]:
    out = ["| 模型 | 格 | 正解格都查到時答對 |", "|---|---|---|"]
    for m in MODELS:
        for c in (c for c in CELLS[m] if c.startswith("fix")):
            x = [
                float(r["right"])
                for r in rows
                if (r["model"], r["condition"]) == (m, c)
                and r["kind"] != "abstain"
                and step1_class(r, questions[r["qid"]]) == "cells_ok"
            ]
            if x:
                mean, lo, hi = bootstrap_ci(x)
                out.append(f"| {m} | {c} | {_r(mean)} [{_r(lo)}, {_r(hi)}]（{len(x)}） |")
    return out


def main() -> int:
    setup_logging("rag_lab")
    rows = [json.loads(ln) for ln in SCORED.read_text("utf-8").splitlines() if ln.strip()]
    last = {}
    for r in read_raw(RAW):
        if not r.get("error"):
            last[(r["model"], r["condition"], r["qid"])] = r
    raw = list(last.values())
    questions = {
        q["id"]: q for q in map(json.loads, QUESTIONS.read_text("utf-8").splitlines()) if q
    }
    questions = {k: q for k, q in questions.items() if not q.get("defect")}  # 2026-11-03
    raw = [r for r in raw if r["qid"] in questions]
    text = "\n".join(
        [
            "# 21 跑完才算的（描述，不下判定）",
            "",
            "## 回傳內容裡有沒有正解格（有答案的題；seg 的第一步跟配對的 fix 相同）",
            "",
            *shown_table(rows, questions),
            "",
            "### 按鏈",
            "",
            *shown_by_chain(rows, questions),
            "",
            "## 第一步在哪一關掉的（19 的 fix，seg 沿用同一份查詢）",
            "",
            *classes(rows, questions),
            "",
            "## 每格兩次（gpt-oss）",
            "",
            *TWICE_HEAD,
            *twice(raw, questions, "gpt-oss:20b", "fix"),
            *twice(raw, questions, "gpt-oss:20b", "seg"),
            "",
        ]
    )
    OUT.write_text(text, "utf-8")
    logger.info("posthoc written", extra={"ctx": {"path": str(OUT), "rows": len(rows)}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
