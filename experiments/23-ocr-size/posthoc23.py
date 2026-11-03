"""After the run: per size, the tables that came back plain, image tokens per page, cell recall,
the 9 questions on 表九二／表四○, and the fallback 'read at s100, re-read at s50 only when a
table came back plain' put together from rows already asked -> results/posthoc.md.

    uv run python experiments/23-ocr-size/posthoc23.py

The fallback needs no new call: each question's evidence is its one gold table, so under the
fallback its prompt is byte for byte the s50 one (plain table) or the s100 one (otherwise).
"""

from __future__ import annotations

import csv
import json
import logging
import statistics
import sys
from pathlib import Path

from rag_lab.log import setup_logging

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from score import cluster_diff_ci, paired_diff_ci  # noqa: E402

logger = logging.getLogger(__name__)

CORPUS = HERE / "corpus"
GLM22 = HERE.parent / "22-degraded-scan/corpus/glm2-L0"  # the same 18 images, read in 22
RESULTS = HERE / "results"
MODELS = ["qwen3:8b", "gpt-oss:20b"]
ROUTES = ["glm2-s100", "glm2-s83", "glm2-s67", "glm2-s50", "glm2-s33"]
WATCH = ["表九二", "表四○"]  # plain at s100 in 06 and 22


def plain_tables(stats: dict, route: str) -> set[str]:
    """Tables with a page that came back as no table at all (markdown or TSV is still one)."""
    return {
        tid
        for tid, t in stats[route].items()
        if any(p["format"] == "plain" for p in t["per_page"].values())
    }


def fallback_rows(rows: list[dict], questions: list[dict], plain: set[str]) -> list[dict]:
    gold = {q["id"]: q["gold"][0] for q in questions}
    pick = {
        (r["model"], r["qid"]): r
        for r in rows
        if r["condition"] == ("glm2-s50" if gold[r["qid"]] in plain else "glm2-s100")
    }
    return [{**r, "condition": "fallback", "from": r["condition"]} for r in pick.values()]


def _ok(rows, model, cond):
    return {
        r["qid"]: float(r["value_ok"])
        for r in rows
        if r["model"] == model and r["condition"] == cond
    }


def main() -> int:
    setup_logging("rag_lab")
    stats = json.loads((CORPUS / "stats.json").read_text("utf-8"))
    questions = [
        json.loads(ln) for ln in (HERE / "questions.jsonl").read_text("utf-8").splitlines()
    ]
    rows = [json.loads(ln) for ln in (RESULTS / "scored.jsonl").read_text("utf-8").splitlines()]
    struct = list(csv.DictReader((RESULTS / "structure.csv").open(encoding="utf-8")))
    gold = {q["id"]: q["gold"][0] for q in questions}
    s100 = CORPUS / "glm2-s100"
    same = [f.name for f in GLM22.iterdir() if (s100 / f.name).read_bytes() == f.read_bytes()]
    out = ["# 23 posthoc", "", "## 雜訊底線", ""]
    out += [f"- glm2-s100 跟 22 的 glm2-L0（同一批圖）逐字相同：{len(same)}/17 張表", ""]
    out += ["## 每一級：回純文字的表、提示 token、格子召回", ""]
    out += ["提示 token 是 `prompt_eval_count`：圖加上提示文字，幾乎全是圖。", ""]
    out += [
        "| 路線 | 圖（寬×高） | 提示 token（中位） | 回純文字 | 格子召回（加權） |",
        "|---|---|---|---|---|",
    ]
    for route in ROUTES:
        pages = [p for t in stats[route].values() for p in t["per_page"].values()]
        mine = [r for r in struct if r["route"] == route]
        n = sum(int(r["n_cells"]) for r in mine)
        cell = sum(float(r["cell_recall"]) * int(r["n_cells"]) for r in mine) / n
        out.append(
            f"| {route} | {pages[0]['size'][0]}×{pages[0]['size'][1]} "
            f"| {statistics.median(p['prompt_eval_count'] for p in pages):.0f} "
            f"| {'、'.join(sorted(plain_tables(stats, route))) or '無'} | {cell:.3f} |"
        )
    out += [
        "",
        "## 格子召回 < 1 的表",
        "",
        "| 路線 | 表 | 格子 | 數字格 | 列 |",
        "|---|---|---|---|---|",
    ]
    for r in struct:
        if r["route"] != "clean" and float(r["cell_recall"]) < 1:
            out.append(
                f"| {r['route']} | {r['table']} | {float(r['cell_recall']):.2f} "
                f"| {float(r['num_recall']):.2f} | {float(r['row_recall']):.2f} |"
            )
    out += ["", f"## {'、'.join(WATCH)} 的題（答對／題數）", ""]
    out += ["| 模型 | " + " | ".join(ROUTES) + " |", "|---|" + "---|" * len(ROUTES)]
    for m in MODELS:
        cells = []
        for c in ROUTES:
            ok = _ok(rows, m, c)
            qs = [q for q in ok if gold[q] in WATCH]
            cells.append(f"{sum(ok[q] for q in qs):.0f}/{len(qs)}")
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    plain = plain_tables(stats, "glm2-s100")
    fb = fallback_rows(rows, questions, plain)
    out += ["", f"## 組合：s100 讀，回純文字的表（{'、'.join(sorted(plain)) or '無'}）改用 s50", ""]
    out += ["| 模型 | s100 | 組合 | 組合 − s100 [按題] | [按表] |", "|---|---|---|---|---|"]
    for m in MODELS:
        a, b = _ok(fb, m, "fallback"), _ok(rows, m, "glm2-s100")
        qids = sorted(b)
        d, lo, hi = paired_diff_ci([a[q] for q in qids], [b[q] for q in qids])
        by: dict[str, list[float]] = {}
        for q in qids:
            by.setdefault(gold[q], []).append(a[q] - b[q])
        _, clo, chi = cluster_diff_ci(by)
        out.append(
            f"| {m} | {statistics.mean(b.values()):.2f} | {statistics.mean(a.values()):.2f} "
            f"| {d:+.2f} [{lo:+.2f}, {hi:+.2f}] | [{clo:+.2f}, {chi:+.2f}] |"
        )
    (RESULTS / "posthoc.md").write_text("\n".join(out) + "\n", "utf-8")
    logger.info("wrote posthoc", extra={"ctx": {"plain_at_s100": sorted(plain)}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
