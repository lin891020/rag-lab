"""After score.py: the tables of 26 -> results/report.md.

    uv run python experiments/26-hosted-retrieval/posthoc26.py

- right (strict, answerable questions) per model and condition, with 17's local models scored by
  the same ruler from 17's stored replies beside them
- the primary and secondary diffs, by chain, right with the gold rows all in the evidence or not
- closed book, abstain questions, undersizing (critical), noise (Haiku's two runs; Opus against
  20, which asked cards and oracle with these prompts), cost
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from rag_lab.log import setup_logging

HERE = Path(__file__).parent
RESULTS = HERE / "results"
RAW20 = HERE.parent / "20-api-ceiling/results/read/raw.jsonl"
sys.path.insert(0, str(HERE))
import claude  # noqa: E402
from run import BASE, HAIKU, MODELS, OPUS, RAW17, load_questions, read_raw  # noqa: E402
from score import CHAINS, _parse, _to_float, score_rows  # noqa: E402

logger = logging.getLogger(__name__)

LOCAL = ("qwen3:8b", "gpt-oss:20b")
TOL = 1e-6  # two values are the same answer


def _answer(content: str) -> tuple[bool, float | None] | None:
    obj, ok, _ = _parse(content or "")
    if not (ok and isinstance(obj, dict)):
        return None
    return obj.get("found") is True, _to_float(obj.get("value"))


def same_answer(a: str, b: str) -> bool:
    """found and value equal (a stopped or unparseable reply is no answer)."""
    x, y = _answer(a), _answer(b)
    if x is None or y is None:
        return False
    if x[0] != y[0]:
        return False
    if not x[0]:
        return True
    return x[1] is not None and y[1] is not None and abs(x[1] - y[1]) <= TOL


def by_hit(scored: list[dict], model: str, cond: str) -> dict[str, tuple[int, int]]:
    """Answerable rows of a cell, split by whether every gold row was in the evidence:
    {"hit": (right, n), "miss": (right, n)}."""
    out = {"hit": [0, 0], "miss": [0, 0]}
    for r in scored:
        if r["model"] == model and r["condition"] == cond and r["kind"] != "abstain":
            k = "hit" if r["rows_all_hit"] else "miss"
            out[k][0] += bool(r["right"])
            out[k][1] += 1
    return {k: (v[0], v[1]) for k, v in out.items()}


def cost(rows: list[dict], model: str, conds: list[str]) -> dict:
    """Every call of these conditions, failed ones too: count, US$, US$ per question asked."""
    rs = [r for r in rows if r["model"] == model and r["condition"] in conds]
    usd = round(
        claude.price(
            model, sum(r["input_tokens"] for r in rs), sum(r["output_tokens"] for r in rs)
        ),
        4,
    )
    keys = {(r["condition"], r["qid"]) for r in rs}
    good = [r for r in rs if not r.get("error")]
    return {
        "calls": len(rs),
        "usd": usd,
        "per_question": round(usd / len(keys), 5) if keys else 0.0,
        "in_mean": round(statistics.mean(r["input_tokens"] for r in good)) if good else 0,
        "out_mean": round(statistics.mean(r["output_tokens"] for r in good)) if good else 0,
    }


def _local_rows(questions: dict[str, dict]) -> list[dict]:
    """17's local replies for these conditions (last good row per key), scored like 26's."""
    last: dict[tuple[str, str, str], dict] = {}
    with gzip.open(RAW17, "rt", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                r = json.loads(ln)
                if r["condition"] in BASE and not r.get("error"):
                    last[(r["model"], r["condition"], r["qid"])] = r
    return score_rows(list(last.values()), questions)


def _f(x: float) -> str:
    return f"{x:.2f}".replace("0.", ".", 1) if x < 1 else "1.0"


def _mean(rows: list[dict], key: str) -> str:
    return _f(sum(bool(r[key]) for r in rows) / len(rows)) if rows else "—"


def main() -> int:
    setup_logging("rag_lab")
    questions = {q["id"]: q for q in load_questions()}
    scored = [json.loads(ln) for ln in (RESULTS / "scored.jsonl").read_text("utf-8").splitlines()]
    raw = read_raw(RESULTS / "raw.jsonl")
    meta = json.loads((RESULTS / "meta.json").read_text("utf-8"))
    summary = list(csv.DictReader((RESULTS / "summary.csv").open(encoding="utf-8")))
    diffs = list(csv.DictReader((RESULTS / "diffs.csv").open(encoding="utf-8")))
    local = _local_rows(questions)
    cell = defaultdict(list)
    for r in scored + local:
        cell[(r["model"], r["condition"])].append(r)
    ans = lambda rs: [r for r in rs if r["kind"] != "abstain"]  # noqa: E731
    out = [
        "# 26 report",
        "",
        f"送出的模型：{', '.join(meta['served_models'])}；anthropic {meta['anthropic']}；"
        f"呼叫 {meta['calls']} 次、失敗 {meta['failed_calls']}、被停 {meta['stopped']}；"
        f"花費 US${meta['spent_usd']}",
        "",
        "## 答對（嚴格，有答案的題；defect 不計）",
        "",
        "| 模型 | " + " | ".join(BASE) + " |",
        "|---|" + "---|" * len(BASE),
    ]
    ci = {
        (r["model"], r["condition"]): r
        for r in summary
        if r["chain"] == "all" and r["metric"] == "right"
    }
    for m in MODELS:
        cells = []
        for c in BASE:
            s = ci.get((m, c))
            cells.append(
                f"{_f(float(s['mean']))} [{_f(float(s['lo']))}, {_f(float(s['hi']))}]" if s else "—"
            )
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    for m in LOCAL:
        out.append(
            f"| {m}（17，同一把尺重算） | "
            + " | ".join(_mean(ans(cell[(m, c)]), "right") for c in BASE)
            + " |"
        )
    n_ans = len(ans(cell[(OPUS, "cards")]))
    out += ["", f"有答案的題 {n_ans}；qwen3／gpt-oss 沒跑閉卷。", ""]

    out += ["## 配對差距（答對嚴格；按題 bootstrap，括號是按 cluster）", ""]
    out += [
        "粗體是跑前宣告的主檢定，只有它們下判定；其餘只描述。",
        "",
        "| 配對 | 模型 | 差距 [按題] | [按 cluster] | 判定 |",
        "|---|---|---|---|---|",
    ]
    for r in diffs:
        if r["metric"] == "right" and r["scope"] == "all":
            both = r["separable"] == "True" and r["separable_cluster"] == "True"
            name = (
                f"{r['a']} − {r['b']}"
                if r["model"] == r["model_b"]
                else f"{r['model']} {r['a']} − {r['model_b']} {r['b']}"
            )
            star = "**" if r["primary"] == "True" else ""
            out.append(
                f"| {star}{name}{star} | {r['model']} | {float(r['diff']):+.2f} [{float(r['lo']):+.2f}, {float(r['hi']):+.2f}] "
                f"| [{float(r['lo_cluster']):+.2f}, {float(r['hi_cluster']):+.2f}] | {('分得出' if both else '分不出') if r['primary'] == 'True' else '（描述）'} |"
            )

    out += ["", "## 按鏈答對（嚴格，有答案的題）", ""]
    out += ["| 模型 | 條件 | " + " | ".join(CHAINS) + " |", "|---|---|" + "---|" * len(CHAINS)]
    for m in (*MODELS, *LOCAL):
        for c in BASE:
            rs = ans(cell[(m, c)])
            if rs:
                out.append(
                    f"| {m} | {c} | "
                    + " | ".join(
                        _mean([r for r in rs if r["chain"] == ch], "right") for ch in CHAINS
                    )
                    + " |"
                )

    out += ["", "## 正解列全進依據時的答對（有答案的題）", ""]
    out += ["| 模型 | 條件 | 全進：對／題 | 不全：對／題 |", "|---|---|---|---|"]
    for m in (*MODELS, *LOCAL):
        for c in BASE[1:-1]:
            h = by_hit(scored + local, m, c)
            if h["hit"][1] + h["miss"][1]:
                out.append(
                    f"| {m} | {c} | {h['hit'][0]}/{h['hit'][1]} | {h['miss'][0]}/{h['miss'][1]} |"
                )

    out += ["", "## 該棄答的題、選小一級（critical）", ""]
    out += ["| 模型 | 條件 | 該棄答有棄答 | 選小一級（選型題） |", "|---|---|---|---|"]
    for m in (*MODELS, *LOCAL):
        for c in BASE:
            rs = cell[(m, c)]
            if rs:
                ab = [r for r in rs if r["kind"] == "abstain"]
                lad = [r for r in rs if r["kind"] == "ladder"]
                out.append(
                    f"| {m} | {c} | {_mean(ab, 'abstain_ok')}（{len(ab)}） | {_mean(lad, 'critical')}（{len(lad)}） |"
                )

    out += ["", "## 雜訊", ""]
    out += ["| 比較 | 條件 | 同答 | 答對 這次 → 另一次 |", "|---|---|---|---|"]
    by = {(r["model"], r["condition"], r["qid"]): r for r in scored}
    for c in BASE:
        qs = [q for (m, cc, q) in by if m == HAIKU and cc == c]
        same = sum(
            same_answer(by[(HAIKU, c, q)]["content"], by[(HAIKU, f"{c}-2", q)]["content"])
            for q in qs
        )
        ok1 = sum(bool(by[(HAIKU, c, q)]["right"]) for q in qs)
        ok2 = sum(bool(by[(HAIKU, f"{c}-2", q)]["right"]) for q in qs)
        out.append(f"| Haiku 兩次 | {c} | {same}/{len(qs)} | {ok1} → {ok2} |")
    rows20: dict[tuple[str, str, str], dict] = {}
    for ln in RAW20.read_text("utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            if (
                r["model"] in MODELS
                and r["condition"] in ("cards", "oracle")
                and not r.get("error")
            ):
                rows20[(r["model"], r["condition"], r["qid"])] = r
    for m in MODELS:
        for c in ("cards", "oracle"):
            qs = [q for (mm, cc, q) in rows20 if mm == m and cc == c and (m, c, q) in by]
            same = sum(
                same_answer(by[(m, c, q)]["content"], rows20[(m, c, q)]["content"]) for q in qs
            )
            out.append(f"| {m} 對 20（C1／C4／C5） | {c} | {same}/{len(qs)} | — |")

    out += ["", "## 花費（含失敗的呼叫；Haiku 兩次合計）", ""]
    out += [
        "| 模型 | 條件 | 呼叫 | US$ | 每次呼叫 US$ | 輸入 token（平均） | 輸出 token（平均） |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in MODELS:
        for c in BASE:
            k = cost(raw, m, [c, f"{c}-2"])
            out.append(
                f"| {m} | {c} | {k['calls']} | {k['usd']:.2f} | {k['per_question']:.4f} | {k['in_mean']} | {k['out_mean']} |"
            )

    (RESULTS / "report.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    logger.info("wrote report", extra={"ctx": {"rows": len(scored)}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
