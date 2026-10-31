"""Numbers computed after the reading track, described only (no verdicts): primary test 1 split
into C1–C3 / C4–C5, `right` given all gold rows in the evidence or not, and how often a model
answered a should-abstain question,
and how far a reused cell asked again (--redo) lands from 16's copy. Reads results/read/scored.jsonl, writes results/posthoc.md.

    uv run python experiments/17-table-retrieval/posthoc.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from rag_lab.log import setup_logging

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from run_read import read_raw  # noqa: E402
from score import bootstrap_ci, judge, paired_diff_ci  # noqa: E402

logger = logging.getLogger(__name__)

SCORED = HERE / "results" / "read" / "scored.jsonl"
RAW = HERE / "results" / "read" / "raw.jsonl"
QUESTIONS = HERE / "questions.jsonl"
OUT = HERE / "results" / "posthoc.md"
MODELS = ["qwen3:8b", "gpt-oss:20b"]
CELLS = [
    "struct@1500",
    "struct@3000-nosub",
    "struct@3000",
    "cards",
    "decomp",
    "decomp@4000",
    "oracle",
]
SPLITS = {"C1–C3": {"C1", "C2", "C3"}, "C4–C5": {"C4", "C5"}}
PAIRS = [("cards", "struct@3000"), ("decomp", "cards"), ("struct@3000", "struct@3000-nosub")]


def _f(x: float) -> str:
    return f"{x:+.2f}".replace("+0.", "+.").replace("-0.", "−.")


def _r(x: float) -> str:
    return f"{x:.2f}".lstrip("0") if x < 1 else "1.0"


def split_diffs(rows: list[dict]) -> list[str]:
    by = {(r["model"], r["condition"], r["qid"]): r for r in rows if r["kind"] != "abstain"}
    out = ["| 模型 | 配對 | 範圍 | 差距 [95%] |", "|---|---|---|---|"]
    for m in MODELS:
        for a, b in PAIRS:
            for name, chains in SPLITS.items():
                qids = sorted(
                    q for (mm, c, q), r in by.items() if mm == m and c == a and r["chain"] in chains
                )
                d, lo, hi = paired_diff_ci(
                    [by[(m, a, q)]["right"] for q in qids], [by[(m, b, q)]["right"] for q in qids]
                )
                out.append(
                    f"| {m} | {a} − {b} | {name}（{len(qids)}） | {_f(d)} [{_f(lo)}, {_f(hi)}] |"
                )
    return out


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


def abstain(rows: list[dict]) -> list[str]:
    out = ["| 模型 | " + " | ".join(CELLS) + " |", "|---|" + "---|" * len(CELLS)]
    for m in MODELS:
        vals = []
        for c in CELLS:
            x = [
                r
                for r in rows
                if r["model"] == m and r["condition"] == c and r["kind"] == "abstain"
            ]
            vals.append(f"{sum(not r['answered'] for r in x)}/{len(x)}")
        out.append(f"| {m} | " + " | ".join(vals) + " |")
    return out


def rerun(raw: list[dict], questions: dict[str, dict]) -> list[str]:
    """A reused cell asked again (run_read --redo) against 16's copy: same model, same prompt."""
    out = [
        "| 模型 | 格 | 原始輸出一字不差 | 答案（found, value）相同 | 答對 重跑 − 16 [95%]（130） | 對錯翻面 |",
        "|---|---|---|---|---|---|",
    ]
    for m in MODELS:
        for c in ("struct@1500", "oracle"):
            mine = [r for r in raw if (r["model"], r["condition"]) == (m, c) and not r.get("error")]
            old = {r["qid"]: r for r in mine if r.get("reused_from")}
            new = {r["qid"]: r for r in mine if not r.get("reused_from")}
            if not new:
                continue

            def ans(r: dict) -> tuple:
                j = judge(r["raw_content"], questions[r["qid"]])
                return j["answered"], j["value"]

            same = sum(old[q]["raw_content"] == new[q]["raw_content"] for q in new)
            same_ans = sum(ans(old[q]) == ans(new[q]) for q in new)
            ids = [q for q in new if questions[q]["expect"]["kind"] != "abstain"]
            a = [judge(new[q]["raw_content"], questions[q])["right"] for q in ids]
            b = [judge(old[q]["raw_content"], questions[q])["right"] for q in ids]
            d, lo, hi = paired_diff_ci(a, b)
            flips = sum(x != y for x, y in zip(a, b, strict=True))
            out.append(
                f"| {m} | {c} | {same}/{len(new)} | {same_ans}/{len(new)} "
                f"| {_f(d)} [{_f(lo)}, {_f(hi)}] | {flips} |"
            )
    return out


def main() -> int:
    setup_logging("rag_lab")
    rows = [json.loads(ln) for ln in SCORED.read_text("utf-8").splitlines() if ln.strip()]
    questions = {
        q["id"]: q for q in map(json.loads, QUESTIONS.read_text("utf-8").splitlines()) if q
    }
    text = "\n".join(
        [
            "# 17 跑完才算的（描述，不下判定）",
            "",
            "## 主檢定拆鏈：C1–C3 對 C4–C5（答對嚴格）",
            "",
            *split_diffs(rows),
            "",
            "## 正解列全進依據時／不全時的答對率",
            "",
            *given_rows(rows),
            "",
            "## 該棄答題有棄答（30 題）",
            "",
            *abstain(rows),
            "",
            "## 沿用格整格重跑（探針沒過的那格）",
            "",
            *rerun(read_raw(RAW), questions),
            "",
        ]
    )
    OUT.write_text(text, "utf-8")
    logger.info("posthoc written", extra={"ctx": {"path": str(OUT), "rows": len(rows)}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
