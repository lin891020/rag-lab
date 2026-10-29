"""results/read/{summary,diffs}.csv (+ results/09-rescored) -> results/report.md
(tables pasted into README section 3).

    uv run python experiments/10-fix-ablation/report.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"
OLD09 = HERE.parent / "09-messy-real" / "results" / "read" / "summary.csv"
TYPE_COLS = ["all", "value", "articles", "which"]
NUMERAL_COLS = ["value:cn", "value:wan", "value:arabic"]
GROUP_TITLE = {
    "add": "加一個（每個修法 − fixed）",
    "drop": "關一個（full − 關掉它）",
    "anchor": "階梯",
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


def render_read(rows: list[dict], *, metric: str, cols: list[str], conditions: list[str]) -> str:
    """One line per (model, condition) in the given order: metric per type column, then the
    share of questions whose evidence held the gold passage."""
    cells: dict[tuple[str, str], dict[str, str]] = {}
    for r in rows:
        key = (r["model"], r["condition"])
        if r["metric"] == metric and r["type"] in cols:
            cells.setdefault(key, {})[r["type"]] = _ci(r)
        elif r["type"] == "all" and r["metric"] == "evidence_has_gold":
            cells.setdefault(key, {})["gold"] = _f(r["mean"])
    head = {
        "all": "全部",
        "value": "數字題",
        "articles": "條號題",
        "which": "哪一份題",
        "value:cn": "中文數字",
        "value:wan": "含「萬」",
        "value:arabic": "阿拉伯數字",
    }
    out = [
        "| 模型 | 條件 | " + " | ".join(head[c] for c in cols) + " | 依據含正解 |",
        "|---|---|" + "---|" * (len(cols) + 1),
    ]
    models = sorted({m for m, _ in cells})
    for m in models:
        for c in conditions:
            ms = cells.get((m, c))
            if ms:
                out.append(
                    f"| {m} | {c} | " + " | ".join(ms.get(k, "") for k in [*cols, "gold"]) + " |"
                )
    return "\n".join(out)


def render_diffs(rows: list[dict], *, model: str, group: str) -> str:
    """Both scores side by side for one model and one ablation group."""
    by: dict[tuple[str, str], dict[str, dict]] = {}
    for r in rows:
        if r["model"] == model and r["group"] == group:
            by.setdefault((r["a"], r["b"]), {})[r["metric"]] = r
    out = ["| 配對 | 嚴格分 | 判定 | 寬鬆分 | 判定 |", "|---|---|---|---|---|"]
    for (a, b), ms in by.items():
        cells = []
        for m in ("value_ok", "value_lenient"):
            r = ms.get(m)
            if r is None:
                cells += ["", ""]
                continue
            sep = "分得出" if r["separable"] == "True" else "分不出"
            cells += [f"{_signed(r['diff'])} [{_signed(r['lo'])}, {_signed(r['hi'])}]", sep]
        out.append(f"| {a} − {b} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def render_rescored(old: list[dict], new: list[dict]) -> str:
    """09's cells: the ruler 09 used vs this one (strict and lenient)."""
    o = {
        (r["model"], r["condition"]): _f(r["mean"])
        for r in old
        if r["type"] == "all" and r["metric"] == "value_ok"
    }
    n: dict[tuple[str, str], dict[str, str]] = {}
    for r in new:
        if r["type"] == "all" and r["metric"] in ("value_ok", "value_lenient"):
            n.setdefault((r["model"], r["condition"]), {})[r["metric"]] = _ci(r)
    out = ["| 模型 | 09 的條件 | 09 的尺 | 新尺嚴格 | 新尺寬鬆 |", "|---|---|---|---|---|"]
    for key in sorted(n):
        out.append(
            f"| {key[0]} | {key[1]} | {o.get(key, '')} | {n[key].get('value_ok', '')} | "
            f"{n[key].get('value_lenient', '')} |"
        )
    return "\n".join(out)


def main() -> int:
    parts = []
    rd = RESULTS / "read"
    if (rd / "summary.csv").exists():
        s = _read(rd / "summary.csv")
        conds = []
        for r in s:
            if r["condition"] not in conds:
                conds.append(r["condition"])
        order = [c for c in _condition_order() if c in conds]
        parts.append(
            f"### 閱讀軌 嚴格分（value_ok）\n\n{render_read(s, metric='value_ok', cols=TYPE_COLS, conditions=order)}\n"
        )
        parts.append(
            f"### 閱讀軌 寬鬆分（value_lenient）\n\n{render_read(s, metric='value_lenient', cols=TYPE_COLS, conditions=order)}\n"
        )
        parts.append(
            f"### 數字題按寫法（嚴格分）\n\n{render_read(s, metric='value_ok', cols=NUMERAL_COLS, conditions=order)}\n"
        )
        if (rd / "diffs.csv").exists():
            d = _read(rd / "diffs.csv")
            for model in sorted({r["model"] for r in d}):
                for group in ("add", "drop", "anchor"):
                    parts.append(
                        f"{GROUP_TITLE[group]}，{model}：\n\n{render_diffs(d, model=model, group=group)}\n"
                    )
    re9 = RESULTS / "09-rescored" / "summary.csv"
    if re9.exists() and OLD09.exists():
        parts.append(f"### 09 用新尺重算\n\n{render_rescored(_read(OLD09), _read(re9))}\n")
    (RESULTS / "report.md").write_text("\n".join(parts), encoding="utf-8")
    return 0


def _condition_order() -> list[str]:
    sys.path.insert(0, str(HERE))
    from prompts import CONDITIONS

    return list(CONDITIONS)


if __name__ == "__main__":
    sys.exit(main())
