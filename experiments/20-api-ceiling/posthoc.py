"""Numbers computed after the reading track, described only (no verdicts): where in the lookup a
reply was lost (step1_class: no lookups / wrong table / wrong row / wrong column / all the gold
cells reached) for Claude and, side by side, 19's small models; how often each class still ended
right; what the lookups returned; each Claude cell asked twice; tokens and cost. Reads
results/read/{scored,raw}.jsonl; writes results/posthoc.md.

    uv run python experiments/20-api-ceiling/posthoc.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

from rag_lab.log import setup_logging

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from run_read import COPIED, RUNS, price, read_raw  # noqa: E402
from score import bootstrap_ci, judge, paired_diff_ci  # noqa: E402

logger = logging.getLogger(__name__)

SCORED = HERE / "results" / "read" / "scored.jsonl"
RAW = HERE / "results" / "read" / "raw.jsonl"
QUESTIONS = HERE / "questions.jsonl"
OUT = HERE / "results" / "posthoc.md"
CELLS = {**COPIED, **RUNS}  # model -> its cells, the small models' copied from 19
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
TOL = 0.01


def _f(x: float) -> str:
    return f"{x:+.2f}".replace("+0.", "+.").replace("-0.", "−.")


def _r(x: float) -> str:
    return f"{x:.2f}".lstrip("0") if x < 1 else "1.0"


def _num(s: str) -> float | None:
    try:
        return float(str(s).replace(",", ""))
    except ValueError:
        return None


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


def tokens(raw: list[dict]) -> list[str]:
    out = [
        "| 模型 | 格 | 讀進 token 最大（一次呼叫） | 讀進合計 | 吐出合計 | 花費 US$ | done_reason |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in RUNS:
        for c in RUNS[m]:
            x = [r for r in raw if (r["model"], r["condition"]) == (m, c) and not r.get("error")]
            if not x:
                continue
            tin = sum(r["input_tokens"] for r in x)
            tout = sum(r["output_tokens"] for r in x)
            peak = max(max(r.get("step1_prompt_eval_count", 0), r["prompt_eval_count"]) for r in x)
            done = Counter(r["done_reason"] for r in x)
            out.append(
                f"| {m} | {c} | {peak} | {tin} | {tout} | {price(m, tin, tout):.2f} "
                f"| {', '.join(f'{k} {v}' for k, v in sorted(done.items()))} |"
            )
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
    twice_rows = [
        ln for m in RUNS for c in ("cards", "oracle", "fix") for ln in twice(raw, questions, m, c)
    ]
    text = "\n".join(
        [
            "# 20 跑完才算的（描述，不下判定）",
            "",
            "## 查表在哪一關掉的（有答案的題；括號是其中答對幾題；小模型是 19 的列）",
            "",
            *classes(rows, questions),
            "",
            "### 按鏈（fix 第一次）",
            "",
            *by_chain_class(rows, questions),
            "",
            "### 正解格都查到時的答對率",
            "",
            *right_when_ok(rows, questions),
            "",
            "## 每次查表的結果",
            "",
            *statuses(raw),
            "",
            "## Claude 每格跑兩次",
            "",
            *TWICE_HEAD,
            *twice_rows,
            *twice(raw, questions, "gpt-oss:20b", "fix"),
            "",
            "## token 與花費（Claude）",
            "",
            *tokens(raw),
            "",
        ]
    )
    OUT.write_text(text, "utf-8")
    logger.info("posthoc written", extra={"ctx": {"path": str(OUT), "rows": len(rows)}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
