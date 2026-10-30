"""results/read/{summary,diffs}.csv + results/attribution.csv -> results/report.md"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"
CONDITIONS = ["L0", "L1-fixed", "L1-struct", "L1-oracle", "L2-struct", "L2-oracle"]
CHAINS = ["C1", "C2", "C3", "C4", "C5"]
HEAD = {
    "right": "答對（嚴格）",
    "right_lenient": "答對（寬鬆）",
    "critical": "選小（critical）",
    "abstain_ok": "該棄答有棄答",
    "hard_answer": "該棄答卻硬答",
    "rows_all_hit": "gold 列全在依據",
    "expr_ok": "算式可求值",
    "parse_ok": "JSON 可解析",
}
CAUSE = {
    "hard_answer": "該棄答卻硬答",
    "wrong_abstain": "誤棄答",
    "closed_book": "閉卷",
    "retrieval_miss": "撈錯列",
    "value_copy": "抄錯值",
    "no_expression": "沒算式",
    "formula": "算式選錯",
    "selection": "選型錯",
    "arithmetic": "算錯",
    "unit_only": "只錯單位",
    "other": "其他",
}


def _f(x) -> str:
    v = round(float(x), 2)
    return "1.0" if v >= 1 else f"{v:.2f}"[1:]


def _s(x) -> str:
    v = float(x)
    return (
        f"{'+' if v >= 0 else '−'}{abs(v):.2f}"[0] + f"{abs(v):.2f}"[1:]
        if False
        else (f"+{v:.2f}" if v >= 0 else f"−{abs(v):.2f}").replace("+0.", "+.").replace("−0.", "−.")
    )


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def overview(summary: list[dict], model: str) -> str:
    cols = [
        "right",
        "right_lenient",
        "critical",
        "abstain_ok",
        "hard_answer",
        "rows_all_hit",
        "expr_ok",
    ]
    out = ["| 條件 | " + " | ".join(HEAD[c] for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for cond in CONDITIONS:
        cells = []
        for c in cols:
            r = next(
                (
                    x
                    for x in summary
                    if x["model"] == model
                    and x["condition"] == cond
                    and x["chain"] == "all"
                    and x["metric"] == c
                ),
                None,
            )
            cells.append(f"{_f(r['mean'])} [{_f(r['lo'])}, {_f(r['hi'])}]" if r else "—")
        out.append(f"| {cond} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def by_chain(summary: list[dict], model: str, metric: str) -> str:
    out = ["| 條件 | " + " | ".join(CHAINS) + " |", "|---|" + "---|" * len(CHAINS)]
    for cond in CONDITIONS:
        cells = []
        for ch in CHAINS:
            r = next(
                (
                    x
                    for x in summary
                    if x["model"] == model
                    and x["condition"] == cond
                    and x["chain"] == ch
                    and x["metric"] == metric
                ),
                None,
            )
            cells.append(f"{_f(r['mean'])} ({r['n']})" if r else "—")
        out.append(f"| {cond} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def diffs_table(diffs: list[dict], model: str, *, primary: bool) -> str:
    out = ["| 配對 | 範圍 | 指標 | 差距 [95%] | 判定 | cluster 判定 |", "|---|---|---|---|---|---|"]
    for d in diffs:
        if d["model"] != model or (d["primary"] == "True") != primary:
            continue
        verdict = "分得出" if d["separable"] == "True" else "分不出"
        vc = "分得出" if d["separable_cluster"] == "True" else "分不出"
        out.append(
            f"| {d['a']} − {d['b']} | {d['scope']}（{d['n']}） | {HEAD.get(d['metric'], d['metric'])} | {_s(d['diff'])} [{_s(d['lo'])}, {_s(d['hi'])}] | {verdict} | {vc} [{_s(d['lo_cluster'])}, {_s(d['hi_cluster'])}] |"
        )
    return "\n".join(out)


def attribution_table(rows: list[dict], model: str) -> str:
    causes = [c for c in CAUSE if any(r["cause"] == c and r["model"] == model for r in rows)]
    out = [
        "| 條件 | " + " | ".join(CAUSE[c] for c in causes) + " |",
        "|---|" + "---|" * len(causes),
    ]
    for cond in CONDITIONS:
        cells = [
            next(
                (
                    r["n"]
                    for r in rows
                    if r["model"] == model and r["condition"] == cond and r["cause"] == c
                ),
                "0",
            )
            for c in causes
        ]
        out.append(f"| {cond} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> int:
    rd = RESULTS / "read"
    if not (rd / "summary.csv").exists():
        return 0
    s = _read(rd / "summary.csv")
    d = _read(rd / "diffs.csv") if (rd / "diffs.csv").exists() else []
    a = _read(RESULTS / "attribution.csv") if (RESULTS / "attribution.csv").exists() else []
    parts = []
    for model in sorted({r["model"] for r in s}):
        parts.append(f"## {model}\n\n### 全部 160 題\n\n{overview(s, model)}\n")
        parts.append(f"### 按鏈，答對（嚴格）\n\n{by_chain(s, model, 'right')}\n")
        parts.append(f"### 按鏈，選小率（選型鏈）\n\n{by_chain(s, model, 'critical')}\n")
        if d:
            parts.append(
                f"### 主檢定（事前宣告，只有這些下判定）\n\n{diffs_table(d, model, primary=True)}\n"
            )
            parts.append(
                f"### 按鏈的差距（描述，不下判定）\n\n{diffs_table(d, model, primary=False)}\n"
            )
        if a:
            parts.append(f"### 錯在哪一步\n\n{attribution_table(a, model)}\n")
    (RESULTS / "report.md").write_text("\n".join(parts), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
