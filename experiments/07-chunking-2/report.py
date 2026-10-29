"""results/summary.csv + diffs.csv -> results/report.md (the tables pasted into README section 3).

uv run python experiments/07-chunking-2/report.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"


def _f(x: str | float) -> str:
    """0.867 -> '.87'; 1.0 -> '1.0'."""
    v = round(float(x), 2)
    return "1.0" if v >= 1 else f"{v:.2f}"[1:]  # .997 would otherwise print as ".00"


def _signed(x: str | float) -> str:
    v = float(x)
    s = _f(abs(v))
    return ("−" if v < 0 else "+") + s


def render_summary(rows: list[dict], *, exp: str, metrics: list[str]) -> str:
    cells: dict[tuple[str, str], dict[str, str]] = {}
    for r in rows:
        if r["exp"] != exp:
            continue
        cells.setdefault((r["chunking"], r["retriever"]), {})[r["metric"]] = (
            f"{_f(r['mean'])} [{_f(r['lo'])}, {_f(r['hi'])}]"
        )
    out = ["| 切法 | 檢索 | " + " | ".join(metrics) + " |", "|---|---|" + "---|" * len(metrics)]
    for (ch, ret), ms in cells.items():
        out.append(f"| {ch} | {ret} | " + " | ".join(ms.get(m, "") for m in metrics) + " |")
    return "\n".join(out)


def render_diffs(rows: list[dict], *, exp: str, retriever: str, metric: str) -> str:
    out = ["| 配對差距 | 差距 [95% 區間] | 判定 |", "|---|---|---|"]
    for r in rows:
        if (r["exp"], r["retriever"], r["metric"]) != (exp, retriever, metric):
            continue
        sep = "分得出" if str(r["separable"]) == "True" else "分不出"
        out.append(
            f"| {r['a']} − {r['b']} | {_signed(r['diff'])} [{_signed(r['lo'])}, {_signed(r['hi'])}] | {sep} |"
        )
    return "\n".join(out)


def main() -> int:
    summary = list(csv.DictReader((RESULTS / "summary.csv").open(encoding="utf-8")))
    diffs = list(csv.DictReader((RESULTS / "diffs.csv").open(encoding="utf-8")))
    meta = json.loads((RESULTS / "meta.json").read_text(encoding="utf-8"))
    metrics = [f"hit@{meta['k']}"] + [f"hit@{b}" for b in meta["budgets"]]
    parts = []
    n = meta["n_questions"]["article"]
    parts.append(
        f"### 條文題（{n} 題）\n\n{render_summary(summary, exp='article', metrics=metrics)}\n"
    )
    head = f"hit@{meta['budgets'][0]}"
    for ret in ("dense", "bm25", "hybrid"):
        table = render_diffs(diffs, exp="article", retriever=ret, metric=head)
        parts.append(f"配對差距（切法之間），{ret}，{head}：\n\n{table}\n")
    for ret in ("hybrid−dense", "hybrid−bm25", "dense−bm25"):
        table = render_diffs(diffs, exp="article", retriever=ret, metric=head)
        parts.append(f"配對差距（檢索之間），{ret}，{head}：\n\n{table}\n")
    (RESULTS / "report.md").write_text("\n".join(parts), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
