"""After score.py: the tables of 29 -> results/report.md.

    uv run python experiments/29-amendment-diff/posthoc29.py

- right per model × condition, overall and per kind (T1 / T2 / T3), with intervals
- the primary diffs (oracle − agent on T2+T3) and the described pairs
- the old-version recall (old_hit: the gold old article in the union of what the searches
  returned) per agent cell and kind, and the cross table hit / miss × right / wrong: the plan's
  "could not find it, or found it and misread it"
- T1 answers with and without evidence (answer_grounded), closed's right T1 questions and how
  many searches agent spent on them (memory against the tool)
- how the model searched: searches per question, refused searches, turns, stops; T2's share of
  「有改」 answers; T3's swapped answers
- every wrong answer in the agent cells with the gold, the answer, and whether the gold was hit
- noise: Haiku's two agent runs (same answer per kind); the drift probe against 27 if it ran
- cost per cell, query samples from Opus agent
"""

from __future__ import annotations

import csv
import json
import logging
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rag_lab.log import setup_logging

HERE = Path(__file__).parent
RESULTS = HERE / "results"
DRIFT = HERE.parents[1] / "data/29-amendment-diff/drift"
RAW27 = HERE.parent / "27-hosted-agentic-search/results/raw.jsonl"
sys.path.insert(0, str(HERE))
import claude  # noqa: E402
from run import HAIKU, KINDS, MODELS, OPUS, RUNS, load_questions, read_raw, usage_of  # noqa: E402
from score import _num, _parse  # noqa: E402

logger = logging.getLogger(__name__)

COLS = RUNS[HAIKU]
AGENT_CELLS = [(m, c) for m in MODELS for c in RUNS[m] if c.startswith("agent")]


def _f(x: float) -> str:
    return f"{x:.2f}".replace("0.", ".", 1) if x < 1 else "1.0"


def _mean(rows: list[dict], key: str) -> str:
    return _f(sum(bool(r.get(key)) for r in rows) / len(rows)) if rows else "—"


def answer_key(content: str, kind: str) -> tuple | None:
    """What the model answered, for the noise comparison; None when it did not answer."""
    obj, ok, _ = _parse(content or "")
    if not (ok and isinstance(obj, dict)):
        return None
    found = obj.get("found") is True
    if kind == "T1":
        return (found, str(obj.get("old_article") or "") if found else "")
    if kind == "T2":
        return (found, obj.get("changed") is True)
    return (found, _num(obj.get("old_value")), _num(obj.get("new_value")))


def cost(rows: list[dict], model: str, cond: str) -> dict:
    rs = [r for r in rows if r["model"] == model and r["condition"] == cond]
    usd = round(sum(claude.price(model, usage_of(r)) for r in rs), 4)
    keys = {r["qid"] for r in rs}
    good = [r for r in rs if not r.get("error")]
    secs = [r["duration"] for r in good if r.get("duration")]
    return {
        "asked": len(rs),
        "usd": usd,
        "per_question": round(usd / len(keys), 5) if keys else 0.0,
        "calls_mean": round(statistics.mean(r.get("api_calls", 1) for r in good), 2) if good else 0,
        "sec_median": round(statistics.median(secs), 1) if secs else 0.0,
    }


def _ci(summary: list[dict], metric: str, kind: str) -> dict[tuple[str, str], str]:
    return {
        (
            r["model"],
            r["condition"],
        ): f"{_f(float(r['mean']))} [{_f(float(r['lo']))}, {_f(float(r['hi']))}]"
        for r in summary
        if r["kind"] == kind and r["metric"] == metric
    }


def _diff_lines(diffs: list[dict], metric: str, scopes: tuple[str, ...]) -> list[str]:
    out = []
    for r in diffs:
        if r["metric"] != metric or r["scope"] not in scopes:
            continue
        both = r["separable"] == "True" and r["separable_cluster"] == "True"
        name = (
            f"{r['a']} − {r['b']}"
            if r["model"] == r["model_b"]
            else f"{r['model']} {r['a']} − {r['model_b']} {r['b']}"
        )
        star = "**" if r["primary"] == "True" else ""
        verdict = ("分得出" if both else "分不出") if star else "（描述）"
        out.append(
            f"| {star}{name}{star} | {r['model']} | {r['scope']} | {r['n']} | {float(r['diff']):+.2f} "
            f"[{float(r['lo']):+.2f}, {float(r['hi']):+.2f}] | [{float(r['lo_cluster']):+.2f}, {float(r['hi_cluster']):+.2f}] | {verdict} |"
        )
    return out


def drift_probe() -> list[str]:
    """27's 20 questions re-asked with 27's own run.py (Haiku agent): same answers as 27's rows."""
    raw = DRIFT / "raw.jsonl"
    if not raw.exists() or not RAW27.exists():
        return ["（沒跑）"]
    now = {r["qid"]: r for r in read_raw(raw) if r["condition"] == "agent" and not r.get("error")}
    then = {
        r["qid"]: r
        for r in read_raw(RAW27)
        if r["condition"] == "agent" and r["model"] == HAIKU and not r.get("error")
    }
    common = sorted(q for q in now if q in then)

    def key(r):
        obj, ok, _ = _parse(r.get("content") or "")
        if not (ok and isinstance(obj, dict)):
            return None
        return (obj.get("found") is True, _num(obj.get("value")))

    same = sum(key(now[q]) == key(then[q]) for q in common)
    return [
        f"Haiku 用 27 的設定重跑 27 的 {len(common)} 題：同答 {same}/{len(common)}（比 found 與 value）。"
    ]


def main() -> int:
    setup_logging("rag_lab")
    scored = [json.loads(ln) for ln in (RESULTS / "scored.jsonl").read_text("utf-8").splitlines()]
    questions = {q["id"]: q for q in load_questions()}
    raw = read_raw(RESULTS / "raw.jsonl")
    meta = json.loads((RESULTS / "meta.json").read_text("utf-8"))
    summary = list(csv.DictReader((RESULTS / "summary.csv").open(encoding="utf-8")))
    diffs = list(csv.DictReader((RESULTS / "diffs.csv").open(encoding="utf-8")))
    cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in scored:
        cell[(r["model"], r["condition"])].append(r)
    n_kind = Counter(r["kind"] for r in cell[(OPUS, "agent")])
    out = [
        "# 29 report",
        "",
        f"送出的模型：{', '.join(meta['served_models'])}；anthropic {meta['anthropic']}；"
        f"題次 {meta['questions_asked']}、API 呼叫 {meta['api_calls']}、"
        f"失敗 {meta['failed_questions']}、被停 {meta['stopped']}；花費 US${meta['spent_usd']}",
        "",
        f"題數：T1 {n_kind['T1']}、T2 {n_kind['T2']}、T3 {n_kind['T3']}（oracle 只跑 T2、T3）。",
        "",
    ]
    for kind in ("all", "T2+T3", *KINDS):
        ci = _ci(summary, "right", kind)
        label = {"all": "全部", "T2+T3": "T2＋T3（主檢定的範圍）"}.get(kind, kind)
        out += [
            f"## 答對：{label}",
            "",
            "| 模型 | " + " | ".join(COLS) + " |",
            "|---|" + "---|" * len(COLS),
        ]
        for m in MODELS:
            out.append(f"| {m} | " + " | ".join(ci.get((m, c), "—") for c in COLS) + " |")
        out.append("")

    out += [
        "## 配對差距：答對（按題 bootstrap，括號是按章）",
        "",
        "粗體是跑前宣告的主檢定，只有它們下判定；其餘只描述。",
        "",
    ]
    out += [
        "| 配對 | 模型 | 範圍 | n | 差距 [按題] | [按章] | 判定 |",
        "|---|---|---|---|---|---|---|",
    ]
    out += _diff_lines(diffs, "right", ("T2+T3", "all"))
    out += [
        "",
        "按題型（只描述）：",
        "",
        "| 配對 | 模型 | 範圍 | n | 差距 [按題] | [按章] | 判定 |",
        "|---|---|---|---|---|---|---|",
    ]
    out += _diff_lines(diffs, "right", tuple(KINDS))

    out += ["", "## 舊版正解撈全率（agent：正解舊條文在搜尋回傳的聯集裡）", ""]
    out += [
        "| 模型 | 條件 | 全部 | T1（有對應的題） | T2 | T3 | 撈到：對／題 | 沒撈到：對／題 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m, c in AGENT_CELLS:
        rs = [r for r in cell[(m, c)] if r.get("old_hit") is not None and r["gold_old_num"]]
        hit = [r for r in rs if r["old_hit"]]
        miss = [r for r in rs if not r["old_hit"]]
        by = [_mean([r for r in rs if r["kind"] == k], "old_hit") for k in KINDS]
        out.append(
            f"| {m} | {c} | {_mean(rs, 'old_hit')} | {by[0]} | {by[1]} | {by[2]} | "
            f"{sum(bool(r['right']) for r in hit)}/{len(hit)} | {sum(bool(r['right']) for r in miss)}/{len(miss)} |"
        )
    out += [
        "",
        "撈全率的差距（只描述）：",
        "",
        "| 配對 | 模型 | 範圍 | n | 差距 [按題] | [按章] | 判定 |",
        "|---|---|---|---|---|---|---|",
    ]
    out += _diff_lines(diffs, "old_hit", ("all",))

    out += ["", "## T1：答出的舊條號在不在查到的標籤裡（記憶 vs 工具）", ""]
    out += [
        "| 模型 | 條件 | 答對且有依據 | 答對但沒依據 | 答錯 | 新增題答對 |",
        "|---|---|---|---|---|---|",
    ]
    for m, c in AGENT_CELLS:
        rs = [r for r in cell[(m, c)] if r["kind"] == "T1"]
        mapped = [r for r in rs if r["gold_old_num"]]
        out.append(
            f"| {m} | {c} | {sum(bool(r['right'] and r['answer_grounded']) for r in mapped)} | "
            f"{sum(bool(r['right'] and not r['answer_grounded']) for r in mapped)} | "
            f"{sum(not r['right'] for r in mapped)} | "
            f"{sum(bool(r['right']) for r in rs if not r['gold_old_num'])}/{sum(1 for r in rs if not r['gold_old_num'])} |"
        )
    out += ["", "閉卷答對的 T1 題（模型記得舊版），同題 agent 查了幾次：", ""]
    for m in MODELS:
        closed_right = sorted(
            r["qid"] for r in cell[(m, "closed")] if r["kind"] == "T1" and r["right"]
        )
        ag = {r["qid"]: r for r in cell[(m, "agent")]}
        items = [
            f"{q}（{ag[q]['n_searches']} 次{'，對' if ag[q]['right'] else '，錯'}）"
            for q in closed_right
            if q in ag
        ]
        out.append(f"- {m}：{len(closed_right)} 題" + ("：" + "、".join(items) if items else ""))

    out += ["", "## 怎麼查的（agent）", ""]
    out += [
        "| 模型 | 條件 | 平均查幾次 | 次數分布 0／1／2／3／4／5 | 被擋的查詢 | 平均輪數 | 被停 | 沒查就答 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m, c in AGENT_CELLS:
        rs = cell[(m, c)]
        dist = Counter(min(r.get("n_searches", 0), 5) for r in rs)
        out.append(
            f"| {m} | {c} | {statistics.mean(r.get('n_searches', 0) for r in rs):.2f} | "
            + "／".join(str(dist[i]) for i in range(6))
            + f" | {sum(r.get('refused_searches', 0) for r in rs)} | {statistics.mean(r.get('turns', 0) for r in rs):.2f} | "
            f"{sum(bool(r.get('stopped')) for r in rs)} | {sum(r.get('n_searches', 0) == 0 for r in rs)} |"
        )

    out += ["", "## T2 答「有改」的比例、T3 新舊對調", ""]
    gold_changed = sum(q["expect"]["changed"] for q in questions.values() if q["kind"] == "T2")
    out += [f"T2 的正解：有改 {gold_changed}、沒改 {n_kind['T2'] - gold_changed}。", ""]
    out += [
        "| 模型 | 條件 | T2 答有改 | T2 答 found false | T3 對調 | T3 found false |",
        "|---|---|---|---|---|---|",
    ]
    for m in MODELS:
        for c in RUNS[m]:
            t2 = [r for r in cell[(m, c)] if r["kind"] == "T2"]
            t3 = [r for r in cell[(m, c)] if r["kind"] == "T3"]
            out.append(
                f"| {m} | {c} | {sum(bool(r.get('answer_changed')) for r in t2)}/{len(t2)} | "
                f"{sum(not r.get('found') for r in t2)}/{len(t2)} | "
                f"{sum(bool(r.get('swapped')) for r in t3)}/{len(t3)} | {sum(not r.get('found') for r in t3)}/{len(t3)} |"
            )

    out += ["", "## 答錯的題（agent）", ""]
    out += [
        "| 模型 | 條件 | 題 | 正解舊條 | 答 | 撈到正解 | 查幾次 |",
        "|---|---|---|---|---|---|---|",
    ]
    for m, c in AGENT_CELLS:
        for r in cell[(m, c)]:
            if r["right"]:
                continue
            if r["kind"] == "T1":
                ans = (
                    ("新增" if not r.get("found") else r.get("answer_old_num") or "?")
                    if r.get("parse_ok")
                    else "（停）"
                )
            elif r["kind"] == "T2":
                ans = (
                    ("有改" if r.get("answer_changed") else "沒改")
                    if r.get("parse_ok")
                    else "（停）"
                )
            else:
                ans = (
                    f"{r.get('answer_old_value')}→{r.get('answer_new_value')}"
                    if r.get("parse_ok")
                    else "（停）"
                )
            out.append(
                f"| {m} | {c} | {r['qid']} | {r['gold_old_num'] or '新增'} | {ans} | "
                f"{'是' if r.get('old_hit') else '否'} | {r.get('n_searches', 0)} |"
            )

    out += ["", "## 雜訊：Haiku agent 兩次", ""]
    a = {r["qid"]: r for r in cell[(HAIKU, "agent")]}
    b = {r["qid"]: r for r in cell[(HAIKU, "agent-2")]}
    out += ["| 題型 | 同答 | 答對 第一次 → 第二次 |", "|---|---|---|"]
    for k in ("all", *KINDS):
        qs = [q for q in a if q in b and (k == "all" or a[q]["kind"] == k)]
        same = sum(
            answer_key(a[q]["content"], a[q]["kind"]) == answer_key(b[q]["content"], a[q]["kind"])
            for q in qs
        )
        out.append(
            f"| {k} | {same}/{len(qs)} | {sum(bool(a[q]['right']) for q in qs)} → {sum(bool(b[q]['right']) for q in qs)} |"
        )

    out += ["", "## 漂移探針", ""] + drift_probe()

    out += ["", "## 花費（每格；含失敗的題次）", ""]
    out += [
        "| 模型 | 條件 | 題次 | US$ | 每題 US$ | 平均 API 呼叫 | 秒（中位） |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in MODELS:
        for c in RUNS[m]:
            k = cost(raw, m, c)
            out.append(
                f"| {m} | {c} | {k['asked']} | {k['usd']:.2f} | {k['per_question']:.4f} | {k['calls_mean']} | {k['sec_median']} |"
            )

    out += [
        "",
        "## 查詢的樣子（Opus agent，每種題前 3 題）",
        "",
        "| 題 | 查詢 | 撈到正解 | 答對 |",
        "|---|---|---|---|",
    ]
    for k in KINDS:
        for r in [r for r in cell[(OPUS, "agent")] if r["kind"] == k][:3]:
            out.append(
                f"| {r['qid']} | {' ／ '.join(r.get('queries', []))} | {r.get('old_hit')} | {r['right']} |"
            )

    (RESULTS / "report.md").write_text("\n".join(out) + "\n", "utf-8")
    logger.info("wrote report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
