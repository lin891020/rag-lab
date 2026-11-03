"""After the run: reply formats per route, which glm2 tables came back byte-identical across
levels (and against 06), structure per table, per-table answers, the questions that flip between
glm2-L0 and L1/L2, and what the opus pages cost -> results/posthoc.md.

    uv run python experiments/22-degraded-scan/posthoc.py
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from rag_lab.log import setup_logging

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from claude_ocr import OCR_MODEL, price  # noqa: E402

logger = logging.getLogger(__name__)

CORPUS = HERE / "corpus"
RESULTS = HERE / "results"
MODELS = ["qwen3:8b", "gpt-oss:20b"]
GLM = ["glm2-L0", "glm2-L1", "glm2-L2"]


def formats(stats: dict) -> dict[str, dict[str, int]]:
    return {
        route: dict(Counter(p["format"] for t in tables.values() for p in t["per_page"].values()))
        for route, tables in stats.items()
    }


def same_files(a: Path, b: Path) -> tuple[list[str], list[str]]:
    """Files of a that b has byte for byte, and those it has different."""
    same, diff = [], []
    for f in sorted(a.iterdir()):
        (same if (b / f.name).read_bytes() == f.read_bytes() else diff).append(f.name)
    return same, diff


def by_table(rows: list[dict], questions: list[dict], model: str, cond: str) -> dict:
    gold = {q["id"]: q["gold"][0] for q in questions}
    out: dict[str, tuple[int, int]] = {}
    for r in rows:
        if r["model"] == model and r["condition"] == cond:
            ok, n = out.get(gold[r["qid"]], (0, 0))
            out[gold[r["qid"]]] = (ok + int(r["value_ok"]), n + 1)
    return out


def flips(rows: list[dict], model: str, a: str, b: str) -> dict[str, list[str]]:
    ok = {(r["condition"], r["qid"]): r["value_ok"] for r in rows if r["model"] == model}
    qids = sorted({q for c, q in ok if c == a})
    return {
        "lost": [q for q in qids if ok[(a, q)] and not ok[(b, q)]],
        "gained": [q for q in qids if not ok[(a, q)] and ok[(b, q)]],
    }


def opus_cost(ledger: Path) -> tuple[int, int, float]:
    rows = [json.loads(ln) for ln in ledger.read_text("utf-8").splitlines() if ln.strip()]
    ti, to = sum(r["tokens_in"] for r in rows), sum(r["tokens_out"] for r in rows)
    return ti, to, price(OCR_MODEL, ti, to)


def main() -> int:
    setup_logging("rag_lab")
    stats = json.loads((CORPUS / "stats.json").read_text("utf-8"))
    questions = [
        json.loads(ln) for ln in (HERE / "questions.jsonl").read_text("utf-8").splitlines()
    ]
    rows = [json.loads(ln) for ln in (RESULTS / "scored.jsonl").read_text("utf-8").splitlines()]
    struct = list(csv.DictReader((RESULTS / "structure.csv").open(encoding="utf-8")))
    out = ["# 22 posthoc", "", "## 回傳格式（頁數）", ""]
    for route, c in formats(stats).items():
        out.append(f"- {route}：" + "、".join(f"{k} {v}" for k, v in sorted(c.items())))
    out += ["", "## glm2 逐字相同的表", ""]
    for a, b in [("glm2-06", "glm2-L0"), ("glm2-L0", "glm2-L1"), ("glm2-L0", "glm2-L2")]:
        same, diff = same_files(CORPUS / a, CORPUS / b)
        out.append(
            f"- {a} vs {b}：{len(same)}/{len(same) + len(diff)} 相同；不同：{'、'.join(f.removesuffix('.txt') for f in diff) or '無'}"
        )
    out += [
        "",
        "## 格子召回（按表，<1 的才列）",
        "",
        "| 路線 | 表 | 格子 | 數字格 | 列 |",
        "|---|---|---|---|---|",
    ]
    for r in struct:
        if r["route"] != "clean" and float(r["cell_recall"]) < 1:
            out.append(
                f"| {r['route']} | {r['table']} | {float(r['cell_recall']):.2f} | {float(r['num_recall']):.2f} | {float(r['row_recall']):.2f} |"
            )
    out += ["", "## 答對（按表，glm2 三級）", ""]
    for m in MODELS:
        tabs = {c: by_table(rows, questions, m, c) for c in GLM}
        out += [f"**{m}**", "", "| 表 | " + " | ".join(GLM) + " |", "|---|" + "---|" * len(GLM)]
        for t in sorted(tabs["glm2-L0"]):
            out.append(f"| {t} | " + " | ".join("{}/{}".format(*tabs[c][t]) for c in GLM) + " |")
        out.append("")
    out += ["## glm2-L0 → L1／L2 翻面的題", ""]
    for m in MODELS:
        for b in GLM[1:]:
            f = flips(rows, m, "glm2-L0", b)
            out.append(
                f"- {m} L0→{b}：掉 {len(f['lost'])}（{'、'.join(f['lost']) or '—'}）；得 {len(f['gained'])}（{'、'.join(f['gained']) or '—'}）"
            )
    ti, to, usd = opus_cost(CORPUS / "opus-ledger.jsonl")
    out += ["", "## opus 讀圖花費", "", f"- 36 頁，輸入 {ti:,}、輸出 {to:,} token，US${usd:.2f}"]
    (RESULTS / "posthoc.md").write_text("\n".join(out) + "\n", "utf-8")
    logger.info("wrote posthoc", extra={"ctx": {"usd": round(usd, 3)}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
