"""results/read/{summary,diffs}.csv -> results/report.md (tables pasted into README section 3).

uv run python experiments/11-abstain/report.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"
KINDS = ["gold", "same-doc", "other-doc", "none", "real", "all"]
HEAD = {
    "answered": "硬答率",
    "answered_f": "濾網後硬答率",
    "lucky": "沒根據的對",
    "grounded": "引句在依據裡",
    "value_ok": "答對（嚴格）",
    "value_lenient": "答對（寬鬆）",
    "value_ok_f": "濾網後答對",
}


def _f(x: str | float) -> str:
    v = round(float(x), 2)
    return "1.0" if v >= 1 else f"{v:.2f}"[1:]  # .997 would otherwise print as ".00"


def _signed(x: str | float) -> str:
    v = float(x)
    return ("−" if v < 0 else "+") + _f(abs(v))


def _ci(r: dict) -> str:
    return f"{_f(r['mean'])} [{_f(r['lo'])}, {_f(r['hi'])}]"


def _read(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def render_group(rows: list[dict], *, group: str, metric: str, qtype: str = "all") -> str:
    """model × prompt down, evidence kind across, one metric."""
    cells: dict[tuple[str, str], dict[str, str]] = {}
    ns: dict[str, str] = {}
    for r in rows:
        if (r["group"], r["metric"], r["type"]) != (group, metric, qtype):
            continue
        cells.setdefault((r["model"], r["prompt"]), {})[r["evidence"]] = _ci(r)
        ns[r["evidence"]] = r["n"]
    kinds = [k for k in KINDS if k in ns]
    out = [
        "| 模型 | 提示 | " + " | ".join(f"{k}（{ns[k]}）" for k in kinds) + " |",
        "|---|---|" + "---|" * len(kinds),
    ]
    for (m, p), ms in sorted(cells.items()):
        out.append(f"| {m} | {p} | " + " | ".join(ms.get(k, "") for k in kinds) + " |")
    return "\n".join(out)


def render_types(rows: list[dict], *, group: str, metric: str) -> str:
    cells: dict[tuple[str, str], dict[str, str]] = {}
    for r in rows:
        if (r["group"], r["metric"], r["evidence"]) != (group, metric, "all"):
            continue
        cells.setdefault((r["model"], r["prompt"]), {})[r["type"]] = _ci(r)
    cols = ["all", "value", "articles", "which"]
    out = ["| 模型 | 提示 | 全部 | 數字題 | 條號題 | 哪一份題 |", "|---|---|---|---|---|---|"]
    for (m, p), ms in sorted(cells.items()):
        out.append(f"| {m} | {p} | " + " | ".join(ms.get(k, "") for k in cols) + " |")
    return "\n".join(out)


def render_diffs(rows: list[dict], *, model: str) -> str:
    out = ["| 組 | 指標 | 配對 | 差距 [95% 區間] | 判定 |", "|---|---|---|---|---|"]
    for r in rows:
        if r["model"] != model:
            continue
        sep = "分得出" if r["separable"] == "True" else "分不出"
        pair = f"{r['a']} − {r['b']}" + (f"（{r['prompt']}）" if r["prompt"] else "")
        out.append(
            f"| {r['group']} | {HEAD.get(r['metric'], r['metric'])} | {pair} | "
            f"{_signed(r['diff'])} [{_signed(r['lo'])}, {_signed(r['hi'])}] | {sep} |"
        )
    return "\n".join(out)


def main() -> int:
    parts = []
    rd = RESULTS / "read"
    if not (rd / "summary.csv").exists():
        return 0
    s = _read(rd / "summary.csv")
    for group, metrics in (
        ("unanswerable", ["answered", "answered_f", "lucky", "grounded"]),
        ("answerable", ["value_ok", "value_lenient", "value_ok_f", "answered", "grounded"]),
    ):
        for m in metrics:
            parts.append(f"### {group}：{HEAD[m]}\n\n{render_group(s, group=group, metric=m)}\n")
        key = "answered" if group == "unanswerable" else "value_ok"
        parts.append(
            f"{group} 按題型（{HEAD[key]}）：\n\n{render_types(s, group=group, metric=key)}\n"
        )
    if (rd / "diffs.csv").exists():
        d = _read(rd / "diffs.csv")
        for model in sorted({r["model"] for r in d}):
            parts.append(f"### 配對差距，{model}\n\n{render_diffs(d, model=model)}\n")
    (RESULTS / "report.md").write_text("\n".join(parts), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
