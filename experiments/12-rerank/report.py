"""results/rerank/{summary,diffs}.csv + results/read/{summary,diffs}.csv -> results/report.md

uv run python experiments/12-rerank/report.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"
RERANKERS = ["none", "bm25", "bge", "qwen3-rr", "listwise"]
RMETRICS = ["hit@1", "mrr", "hit@5", "hit@1000", "hit@2000", "budget_has_gold"]
HEAD = {
    "hit@1": "hit@1",
    "mrr": "MRR",
    "hit@5": "hit@5",
    "hit@1000": "hit@1000",
    "hit@2000": "hit@2000",
    "budget_has_gold": "預算內有正解",
    "seconds": "秒/題",
}


def _f(x: str | float) -> str:
    v = float(x)
    return "1.0" if v >= 1 else f"{v:.2f}"[1:]


def _signed(x: str | float) -> str:
    v = float(x)
    return ("−" if v < 0 else "+") + _f(abs(v))


def _ci(r: dict) -> str:
    return f"{_f(r['mean'])} [{_f(r['lo'])}, {_f(r['hi'])}]"


def _read(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def render_rerank(rows: list[dict], *, cell: str) -> str:
    cells: dict[str, dict[str, str]] = {}
    for r in rows:
        if r["cell"] != cell:
            continue
        if r["metric"] == "seconds":
            cells.setdefault(r["reranker"], {})["seconds"] = f"{float(r['mean']):.1f}"
        else:
            cells.setdefault(r["reranker"], {})[r["metric"]] = _ci(r)
    cols = [*RMETRICS, "seconds"]
    out = ["| 重排器 | " + " | ".join(HEAD[c] for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for rr in RERANKERS:
        if rr in cells:
            out.append(f"| {rr} | " + " | ".join(cells[rr].get(c, "") for c in cols) + " |")
    return "\n".join(out)


def render_rerank_diffs(rows: list[dict], *, cell: str) -> str:
    cells: dict[str, dict[str, str]] = {}
    for r in rows:
        if r["cell"] != cell:
            continue
        sep = "✓" if r["separable"] == "True" else ""
        cells.setdefault(r["reranker"], {})[r["metric"]] = (
            f"{_signed(r['diff'])} [{_signed(r['lo'])}, {_signed(r['hi'])}]{sep}"
        )
    out = [
        "| 重排器 − none | " + " | ".join(HEAD[c] for c in RMETRICS) + " |",
        "|---|" + "---|" * len(RMETRICS),
    ]
    for rr in RERANKERS:
        if rr in cells:
            out.append(f"| {rr} | " + " | ".join(cells[rr].get(c, "") for c in RMETRICS) + " |")
    return "\n".join(out)


def render_read(rows: list[dict], *, metric: str) -> str:
    cells: dict[tuple[str, str], dict[str, str]] = {}
    for r in rows:
        key = (r["model"], r["condition"])
        if r["metric"] == metric and r["type"] in ("all", "value", "articles", "which"):
            cells.setdefault(key, {})[r["type"]] = _ci(r)
        elif r["type"] == "all" and r["metric"] == "evidence_has_gold":
            cells.setdefault(key, {})["gold"] = _f(r["mean"])
    out = [
        "| 模型 | 重排器 | 全部 | 數字題 | 條號題 | 哪一份題 | 依據含正解 |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in sorted({m for m, _ in cells}):
        for c in RERANKERS:
            ms = cells.get((m, c))
            if ms:
                out.append(
                    f"| {m} | {c} | "
                    + " | ".join(
                        ms.get(k, "") for k in ("all", "value", "articles", "which", "gold")
                    )
                    + " |"
                )
    return "\n".join(out)


def render_read_diffs(rows: list[dict], *, model: str) -> str:
    by: dict[str, dict[str, dict]] = {}
    for r in rows:
        if r["model"] == model:
            by.setdefault(r["a"], {})[r["metric"]] = r
    out = ["| 配對 | 嚴格分 | 判定 | 寬鬆分 | 判定 |", "|---|---|---|---|---|"]
    for a, ms in by.items():
        cells = []
        for m in ("value_ok", "value_lenient"):
            r = ms.get(m)
            if r is None:
                cells += ["", ""]
                continue
            sep = "分得出" if r["separable"] == "True" else "分不出"
            cells += [f"{_signed(r['diff'])} [{_signed(r['lo'])}, {_signed(r['hi'])}]", sep]
        out.append(f"| {a} − none | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> int:
    parts = []
    rr = RESULTS / "rerank"
    if (rr / "summary.csv").exists():
        s = _read(rr / "summary.csv")
        d = _read(rr / "diffs.csv") if (rr / "diffs.csv").exists() else []
        for cell in ("full", "fixed"):
            if any(r["cell"] == cell for r in s):
                parts.append(f"### 檢索軌 {cell}\n\n{render_rerank(s, cell=cell)}\n")
                if d:
                    parts.append(
                        f"配對差距 {cell}（✓ = 分得出）：\n\n{render_rerank_diffs(d, cell=cell)}\n"
                    )
    rd = RESULTS / "read"
    if (rd / "summary.csv").exists():
        s = _read(rd / "summary.csv")
        parts.append(f"### 閱讀軌 嚴格分\n\n{render_read(s, metric='value_ok')}\n")
        parts.append(f"### 閱讀軌 寬鬆分\n\n{render_read(s, metric='value_lenient')}\n")
        if (rd / "diffs.csv").exists():
            d = _read(rd / "diffs.csv")
            for model in sorted({r["model"] for r in d}):
                parts.append(f"配對差距，{model}：\n\n{render_read_diffs(d, model=model)}\n")
    (RESULTS / "report.md").write_text("\n".join(parts), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
