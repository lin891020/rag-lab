"""results/{retrieval,read}/summary.csv + diffs.csv + attribution.csv -> results/report.md
(tables pasted into README section 3).

    uv run python experiments/09-messy-real/report.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"
RETRIEVERS = ["dense", "bm25", "random"]


def _f(x: str | float) -> str:
    v = round(float(x), 2)
    return "1.0" if v >= 1 else f"{v:.2f}"[1:]  # .997 would otherwise print as ".00"


def _signed(x: str | float) -> str:
    v = float(x)
    return ("−" if v < 0 else "+") + _f(abs(v))


def _ci(r: dict) -> str:
    return f"{_f(r['mean'])} [{_f(r['lo'])}, {_f(r['hi'])}]"


def render_retrieval(rows: list[dict], *, metric: str) -> str:
    """One line per (route, chunking, prefix): dense / bm25 / random with intervals."""
    cells: dict[tuple[str, str, str], dict[str, str]] = {}
    for r in rows:
        if r["metric"] != metric:
            continue
        key = (r["route"], r["chunking"], "有" if r["prefix"] == "True" else "無")
        cells.setdefault(key, {})[r["retriever"]] = _ci(r)
    out = ["| 抽字 | 切塊 | 前綴 | dense | bm25 | random |", "|---|---|---|---|---|---|"]
    for (route, ch, pre), ms in sorted(cells.items()):
        out.append(
            f"| {route} | {ch} | {pre} | " + " | ".join(ms.get(k, "") for k in RETRIEVERS) + " |"
        )
    return "\n".join(out)


def render_retrieval_diffs(rows: list[dict], *, against: str, metric: str, retriever: str) -> str:
    out = ["| 格 | 差距 [95% 區間] | 判定 |", "|---|---|---|"]
    for r in rows:
        if (r["against"], r["metric"], r["retriever"]) != (against, metric, retriever):
            continue
        pre = "有" if r["prefix"] == "True" else "無"
        sep = "分得出" if r["separable"] == "True" else "分不出"
        out.append(
            f"| {r['route']}/{r['chunking']}/前綴{pre} | {_signed(r['diff'])} [{_signed(r['lo'])}, {_signed(r['hi'])}] | {sep} |"
        )
    return "\n".join(out)


def render_read(rows: list[dict]) -> str:
    cells: dict[tuple[str, str], dict[str, str]] = {}
    for r in rows:
        key = (r["model"], r["condition"])
        if r["type"] == "all" and r["metric"] == "value_ok":
            cells.setdefault(key, {})["all"] = _ci(r)
        elif r["type"] != "all" and r["metric"] == "value_ok":
            cells.setdefault(key, {})[r["type"]] = _ci(r)
        elif r["type"] == "all" and r["metric"] == "evidence_has_gold":
            cells.setdefault(key, {})["gold"] = _f(r["mean"])
    cols = ["all", "value", "articles", "which", "gold"]
    out = [
        "| 模型 | 條件 | 全部 | 數字題 | 條號題 | 哪一份題 | 依據含正解 |",
        "|---|---|---|---|---|---|---|",
    ]
    for (m, c), ms in cells.items():
        out.append(f"| {m} | {c} | " + " | ".join(ms.get(k, "") for k in cols) + " |")
    return "\n".join(out)


def render_read_diffs(rows: list[dict], *, model: str) -> str:
    out = ["| 配對差距 | 差距 [95% 區間] | 判定 |", "|---|---|---|"]
    for r in rows:
        if r["model"] != model:
            continue
        sep = "分得出" if r["separable"] == "True" else "分不出"
        out.append(
            f"| {r['a']} − {r['b']} | {_signed(r['diff'])} [{_signed(r['lo'])}, {_signed(r['hi'])}] | {sep} |"
        )
    return "\n".join(out)


def render_attribution(rows: list[dict]) -> str:
    out = ["| 格 | 錯因 | 題數 | 08 預測到？ |", "|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| {r['cell']} | {r['cause']} | {r['n']} | {'是' if r['predicted_by_08'] == 'True' else '否'} |"
        )
    return "\n".join(out)


def main() -> int:
    parts = []
    ret = RESULTS / "retrieval"
    if (ret / "summary.csv").exists():
        s = list(csv.DictReader((ret / "summary.csv").open(encoding="utf-8")))
        d = list(csv.DictReader((ret / "diffs.csv").open(encoding="utf-8")))
        parts.append(f"### 檢索軌 hit@1000\n\n{render_retrieval(s, metric='hit@1000')}\n")
        parts.append(f"### 檢索軌 hit@5\n\n{render_retrieval(s, metric='hit@5')}\n")
        for against in ("base/article/noprefix", "same/noprefix", "base/same", "same/article"):
            parts.append(
                f"配對差距（對 {against}，dense，hit@1000）：\n\n"
                f"{render_retrieval_diffs(d, against=against, metric='hit@1000', retriever='dense')}\n"
            )
    att = RESULTS / "attribution.csv"
    if att.exists():
        parts.append(
            f"### 錯因\n\n{render_attribution(list(csv.DictReader(att.open(encoding='utf-8'))))}\n"
        )
    rd = RESULTS / "read"
    if (rd / "summary.csv").exists():
        s = list(csv.DictReader((rd / "summary.csv").open(encoding="utf-8")))
        d = list(csv.DictReader((rd / "diffs.csv").open(encoding="utf-8")))
        parts.append(f"### 閱讀軌 value_ok\n\n{render_read(s)}\n")
        for model in sorted({r["model"] for r in d}):
            parts.append(f"配對差距，{model}：\n\n{render_read_diffs(d, model=model)}\n")
    (RESULTS / "report.md").write_text("\n".join(parts), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
