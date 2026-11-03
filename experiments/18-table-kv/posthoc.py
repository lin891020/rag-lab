"""Numbers computed after the reading track, described only (no verdicts): `right` given all
gold rows in the evidence or not, which cell a wrong reply read (attribute.read_slip), how far
gpt-oss's cards asked again in 18 lands from 17's copy (and 17's test 1 recomputed with each),
and the prompt tokens per cell (a cut prompt would show up here). Reads results/read/scored.jsonl,
results/read/raw.jsonl and 17's raw.jsonl.gz; writes results/posthoc.md.

    uv run python experiments/18-table-kv/posthoc.py
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
from attribute import read_slip  # noqa: E402
from run_read import OPTIONS, read_raw, rows17  # noqa: E402
from score import bootstrap_ci, judge, paired_diff_ci  # noqa: E402

logger = logging.getLogger(__name__)

SCORED = HERE / "results" / "read" / "scored.jsonl"
RAW = HERE / "results" / "read" / "raw.jsonl"
QUESTIONS = HERE / "questions.jsonl"
OUT = HERE / "results" / "posthoc.md"
MODELS = ["qwen3:8b", "gpt-oss:20b"]
CELLS = ["cards", "cards-kv"]
SPLITS = {"C1–C3": {"C1", "C2", "C3"}, "C4–C5": {"C4", "C5"}}
SLIPS = ["wrong_column", "wrong_row", "other_number"]


def _f(x: float) -> str:
    return f"{x:+.2f}".replace("+0.", "+.").replace("-0.", "−.")


def _r(x: float) -> str:
    return f"{x:.2f}".lstrip("0") if x < 1 else "1.0"


def given_rows(rows: list[dict]) -> list[str]:
    out = [
        "| 模型 | 格 | 正解列全進時答對 | 不全時 | 全進時、只看 C1–C3 |",
        "|---|---|---|---|---|",
    ]
    for m in MODELS:
        for c in CELLS:
            x = [
                r
                for r in rows
                if r["model"] == m and r["condition"] == c and r["kind"] != "abstain"
            ]
            hit = [r for r in x if r["rows_all_hit"]]
            miss = [r for r in x if not r["rows_all_hit"]]
            hit3 = [r for r in hit if r["chain"] in SPLITS["C1–C3"]]

            def cell(z: list[dict]) -> str:
                if not z:
                    return "—"
                mean, lo, hi = bootstrap_ci([float(r["right"]) for r in z])
                return f"{_r(mean)} [{_r(lo)}, {_r(hi)}]（{len(z)}）"

            out.append(f"| {m} | {c} | {cell(hit)} | {cell(miss)} | {cell(hit3)} |")
    return out


def latest(rows: list[dict]) -> dict[tuple[str, str, str], dict]:
    """The last good row per (model, cell, qid), as score.py keeps them."""
    out = {}
    for r in rows:
        if not r.get("error"):
            out[(r["model"], r["condition"], r["qid"])] = r
    return out


def slips(rows: list[dict], questions: dict[str, dict]) -> list[str]:
    out = [
        "| 模型 | 格 | 答錯但正解列全在 | " + " | ".join(SLIPS) + " |",
        "|---|---|---|" + "---|" * len(SLIPS),
    ]
    for m in MODELS:
        for c in CELLS:
            x = [r for r in rows if r["model"] == m and r["condition"] == c]
            got = Counter(read_slip(r, questions[r["qid"]]) for r in x)
            eligible = sum(got[k] for k in SLIPS)
            out.append(
                f"| {m} | {c} | {eligible} | " + " | ".join(str(got[k]) for k in SLIPS) + " |"
            )
    return out


def rerun(
    old: list[dict], new: list[dict], questions: dict[str, dict], model: str, cell: str
) -> list[str]:
    """The same (model, cell) asked in 17 and again in 18: same model digest, same prompt."""
    a17 = {r["qid"]: r for r in old if (r["model"], r["condition"]) == (model, cell)}
    a18 = {
        r["qid"]: r
        for r in new
        if (r["model"], r["condition"]) == (model, cell) and not r.get("reused_from")
    }
    qids = sorted(set(a17) & set(a18))
    if not qids:
        return ["（沒有重跑的列）"]

    def ans(r: dict) -> tuple:
        j = judge(r["raw_content"], questions[r["qid"]])
        return j["answered"], j["value"]

    same = sum(a17[q]["raw_content"] == a18[q]["raw_content"] for q in qids)
    same_ans = sum(ans(a17[q]) == ans(a18[q]) for q in qids)
    ids = [q for q in qids if questions[q]["expect"]["kind"] != "abstain"]
    x = [judge(a18[q]["raw_content"], questions[q])["right"] for q in ids]
    y = [judge(a17[q]["raw_content"], questions[q])["right"] for q in ids]
    d, lo, hi = paired_diff_ci(x, y)
    flips = sum(p != q for p, q in zip(x, y, strict=True))
    return [
        f"| 模型 | 格 | 原始輸出一字不差 | 答案（found, value）相同 | 答對 18 − 17 [95%]（{len(ids)}） | 對錯翻面 |",
        "|---|---|---|---|---|---|",
        f"| {model} | {cell} | {same}/{len(qids)} | {same_ans}/{len(qids)} "
        f"| {_f(d)} [{_f(lo)}, {_f(hi)}] | {flips} |",
    ]


def test1_again(
    old: list[dict], new: list[dict], questions: dict[str, dict], model: str
) -> list[str]:
    """17's test 1 (cards − struct@3000) with 17's cards and with 18's cards."""
    base = {r["qid"]: r for r in old if (r["model"], r["condition"]) == (model, "struct@3000")}
    out = ["| cards 用哪一份 | cards − struct@3000 [95%] |", "|---|---|"]
    for name, rows in (
        ("17", [r for r in old if (r["model"], r["condition"]) == (model, "cards")]),
        (
            "18",
            [
                r
                for r in new
                if (r["model"], r["condition"]) == (model, "cards") and not r.get("reused_from")
            ],
        ),
    ):
        got = {r["qid"]: r for r in rows}
        ids = sorted(q for q in got if q in base and questions[q]["expect"]["kind"] != "abstain")
        if not ids:
            continue
        d, lo, hi = paired_diff_ci(
            [judge(got[q]["raw_content"], questions[q])["right"] for q in ids],
            [judge(base[q]["raw_content"], questions[q])["right"] for q in ids],
        )
        out.append(f"| {name} | {_f(d)} [{_f(lo)}, {_f(hi)}]（{len(ids)}） |")
    return out


def tokens(raw: list[dict]) -> list[str]:
    limit = OPTIONS["num_ctx"]
    out = [
        f"| 模型 | 格 | 提示字數 最大 | prompt_eval_count 最大（上限 {limit}） | done_reason |",
        "|---|---|---|---|---|",
    ]
    for m in MODELS:
        for c in CELLS:
            x = [
                r
                for r in raw
                if r["model"] == m and r["condition"] == c and not r.get("reused_from")
            ]
            if not x:
                continue
            pec = max(r.get("prompt_eval_count", 0) for r in x)
            reasons = Counter(r["done_reason"] for r in x)
            out.append(
                f"| {m} | {c} | {max(r['prompt_len'] for r in x)} | {pec} "
                f"| {', '.join(f'{k} {v}' for k, v in sorted(reasons.items()))} |"
            )
    return out


def main() -> int:
    setup_logging("rag_lab")
    rows = list(
        latest(
            [json.loads(ln) for ln in SCORED.read_text("utf-8").splitlines() if ln.strip()]
        ).values()
    )
    raw = read_raw(RAW)
    old = rows17()
    questions = {
        q["id"]: q for q in map(json.loads, QUESTIONS.read_text("utf-8").splitlines()) if q
    }
    questions = {k: q for k, q in questions.items() if not q.get("defect")}  # 2026-11-03
    raw = [r for r in raw if r["qid"] in questions]
    old = [r for r in old if r["qid"] in questions]
    text = "\n".join(
        [
            "# 18 跑完才算的（描述，不下判定）",
            "",
            "## 正解列全進依據時／不全時的答對率",
            "",
            *given_rows(rows),
            "",
            "## 答錯的時候讀了哪一格（正解列全在依據裡的才算）",
            "",
            *slips(rows, questions),
            "",
            "## gpt-oss 的 cards：17 跑一次、18 再跑一次",
            "",
            *rerun(old, raw, questions, "gpt-oss:20b", "cards"),
            "",
            "### 17 的檢定 1 用兩份 cards 各算一次",
            "",
            *test1_again(old, raw, questions, "gpt-oss:20b"),
            "",
            "## 提示長度與 token（看有沒有被截）",
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
