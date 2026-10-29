"""results/summary.csv + diffs.csv -> results/report.md (tables pasted into README section 3).

uv run python experiments/06-ocr-source/report.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"
METRICS = ["parse_ok", "schema_ok", "value_ok", "value_only_ok"]


def _f(x: str | float) -> str:
    v = round(float(x), 2)
    return "1.0" if v >= 1 else f"{v:.2f}"[1:]  # .997 would otherwise print as ".00"


def _signed(x: str | float) -> str:
    v = float(x)
    return ("−" if v < 0 else "+") + _f(abs(v))


def render_summary(rows: list[dict], *, metrics: list[str]) -> str:
    cells: dict[tuple[str, str], dict[str, str]] = {}
    for r in rows:
        key = (r["model"], r["condition"])
        if r["metric"] in metrics:
            cells.setdefault(key, {})[r["metric"]] = (
                f"{_f(r['mean'])} [{_f(r['lo'])}, {_f(r['hi'])}]"
            )
        elif r["metric"] in ("eval_count", "seconds"):
            cells.setdefault(key, {})[r["metric"]] = (
                f"{float(r['mean']):.0f}"
                if r["metric"] == "eval_count"
                else f"{float(r['mean']):.1f}"
            )
    cols = metrics + ["eval_count", "seconds"]
    out = ["| 模型 | 條件 | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for (m, c), ms in cells.items():
        out.append(f"| {m} | {c} | " + " | ".join(ms.get(k, "") for k in cols) + " |")
    return "\n".join(out)


def render_diffs(rows: list[dict], *, model: str, metric: str) -> str:
    out = ["| 配對差距 | 差距 [95% 區間] | 判定 |", "|---|---|---|"]
    for r in rows:
        if (r["model"], r["metric"]) != (model, metric):
            continue
        sep = "分得出" if str(r["separable"]) == "True" else "分不出"
        out.append(
            f"| {r['a']} − {r['b']} | {_signed(r['diff'])} [{_signed(r['lo'])}, {_signed(r['hi'])}] | {sep} |"
        )
    return "\n".join(out)


def main() -> int:
    summary = list(csv.DictReader((RESULTS / "summary.csv").open(encoding="utf-8")))
    diffs = list(csv.DictReader((RESULTS / "diffs.csv").open(encoding="utf-8")))
    meta = json.loads((RESULTS / "meta.json").read_text("utf-8"))
    parts = [
        f"### 全部（{meta['n_questions']['total']} 題）\n\n{render_summary(summary, metrics=METRICS)}\n"
    ]
    for model in meta["models"]:
        parts.append(
            f"配對差距，{model}，value_ok：\n\n{render_diffs(diffs, model=model, metric='value_ok')}\n"
        )
    (RESULTS / "report.md").write_text("\n".join(parts), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
