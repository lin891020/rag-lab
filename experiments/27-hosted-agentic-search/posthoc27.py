"""After score.py: the tables of 27 -> results/report.md.

    uv run python experiments/27-hosted-agentic-search/posthoc27.py

- right (strict, answerable questions) and gold rows all in (the union of what the searches
  returned) per model, for agent beside 26's cards, decomp@4000 and oracle
- the primary and secondary diffs (right and rows_all_hit), by chain, right with the gold rows all
  in or not
- how the model searched: searches per question, refused searches, turns, stops, answers given
  without a search, and whether the quote is in the text the searches returned (recomputed from
  the stored queries: the tool is stateless)
- answers given and how many were right, abstain questions, undersizing (critical), noise
  (Haiku's two runs), cost against 26's
"""

from __future__ import annotations

import csv
import json
import logging
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from rag_lab.log import setup_logging

HERE = Path(__file__).parent
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE))
import claude  # noqa: E402
import run  # noqa: E402
from run import HAIKU, MODELS, OPUS, RUNS, read_raw, usage_of  # noqa: E402
from score import BASELINE, CHAINS, RAW26, _parse, _to_float  # noqa: E402

logger = logging.getLogger(__name__)

TOL = 1e-6  # two values are the same answer
CELLS = {HAIKU: ["agent", "agent-2", *BASELINE], OPUS: ["agent", *BASELINE]}
_WS = re.compile(r"\s+")


def _answer(content: str) -> tuple[bool, float | None] | None:
    obj, ok, _ = _parse(content or "")
    if not (ok and isinstance(obj, dict)):
        return None
    return obj.get("found") is True, _to_float(obj.get("value"))


def same_answer(a: str, b: str) -> bool:
    """found and value equal (a stopped or unparseable reply is no answer)."""
    x, y = _answer(a), _answer(b)
    if x is None or y is None or x[0] != y[0]:
        return False
    if not x[0]:
        return True
    return x[1] is not None and y[1] is not None and abs(x[1] - y[1]) <= TOL


def _squash(s: str) -> str:
    return _WS.sub("", unicodedata.normalize("NFKC", s))


def quote_in(quote: str, texts: list[str]) -> bool:
    """The quote, spaces and width folded, is a substring of what the searches returned."""
    q = _squash(quote)
    return bool(q) and any(q in _squash(t) for t in texts)


_NUM = re.compile(r"\d+(?:\.\d+)?")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3})")


def _numbers(s: str) -> set[str]:
    """The numbers in a text, width folded and thousands commas dropped (20,000 -> 20000);
    spaces are kept as separators, so a column name and its value stay two numbers."""
    return set(_NUM.findall(_THOUSANDS.sub("", unicodedata.normalize("NFKC", s))))


def numbers_in(quote: str, texts: list[str]) -> bool:
    """Every number in the quote is a number in what the searches returned: the models rewrite a
    table row into one line (header: values), so the word-for-word check undercounts; this one
    asks whether a number came from nowhere. Added after 27's run, description only."""
    nums = _numbers(quote)
    return bool(nums) and nums <= _numbers("\n".join(texts))


def by_hit(scored: list[dict], model: str, cond: str) -> dict[str, tuple[int, int]]:
    out = {"hit": [0, 0], "miss": [0, 0]}
    for r in scored:
        if r["model"] == model and r["condition"] == cond and r["kind"] != "abstain":
            k = "hit" if r["rows_all_hit"] else "miss"
            out[k][0] += bool(r["right"])
            out[k][1] += 1
    return {k: (v[0], v[1]) for k, v in out.items()}


def cost(rows: list[dict], model: str, cond: str) -> dict:
    """Every question of a cell ever asked, failed ones too: US$, per question, turns, seconds."""
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


def _f(x: float) -> str:
    return f"{x:.2f}".replace("0.", ".", 1) if x < 1 else "1.0"


def _mean(rows: list[dict], key: str) -> str:
    return _f(sum(bool(r[key]) for r in rows) / len(rows)) if rows else "—"


def _ci(summary: list[dict], metric: str) -> dict[tuple[str, str], str]:
    return {
        (
            r["model"],
            r["condition"],
        ): f"{_f(float(r['mean']))} [{_f(float(r['lo']))}, {_f(float(r['hi']))}]"
        for r in summary
        if r["chain"] == "all" and r["metric"] == metric
    }


def _diff_lines(diffs: list[dict], metric: str) -> list[str]:
    out = []
    for r in diffs:
        if r["metric"] == metric and r["scope"] == "all":
            both = r["separable"] == "True" and r["separable_cluster"] == "True"
            name = (
                f"{r['a']} − {r['b']}"
                if r["model"] == r["model_b"]
                else f"{r['model']} {r['a']} − {r['model_b']} {r['b']}"
            )
            star = "**" if r["primary"] == "True" and metric == "right" else ""
            verdict = ("分得出" if both else "分不出") if star else "（描述）"
            out.append(
                f"| {star}{name}{star} | {r['model']} | {float(r['diff']):+.2f} "
                f"[{float(r['lo']):+.2f}, {float(r['hi']):+.2f}] "
                f"| [{float(r['lo_cluster']):+.2f}, {float(r['hi_cluster']):+.2f}] | {verdict} |"
            )
    return out


def main() -> int:
    setup_logging("rag_lab")
    scored = [json.loads(ln) for ln in (RESULTS / "scored.jsonl").read_text("utf-8").splitlines()]
    raw = read_raw(RESULTS / "raw.jsonl")
    raw26 = [r for r in read_raw(RAW26) if r["condition"] in BASELINE]
    meta = json.loads((RESULTS / "meta.json").read_text("utf-8"))
    summary = list(csv.DictReader((RESULTS / "summary.csv").open(encoding="utf-8")))
    diffs = list(csv.DictReader((RESULTS / "diffs.csv").open(encoding="utf-8")))
    cell = defaultdict(list)
    for r in scored:
        cell[(r["model"], r["condition"])].append(r)
    ans = lambda rs: [r for r in rs if r["kind"] != "abstain"]  # noqa: E731
    out = [
        "# 27 report",
        "",
        f"送出的模型：{', '.join(meta['served_models'])}；anthropic {meta['anthropic']}；"
        f"題次 {meta['questions_asked']}、API 呼叫 {meta['api_calls']}、"
        f"失敗 {meta['failed_questions']}、被停 {meta['stopped']}；花費 US${meta['spent_usd']}",
        "",
        "基線是 26 的列（同模型、同一天、同一把尺重算）。",
        "",
    ]
    for title, metric in (
        ("答對（嚴格，有答案的題；defect 不計）", "right"),
        ("正解列全進依據（有答案的題；agent 取所有搜尋回傳的聯集）", "rows_all_hit"),
    ):
        ci = _ci(summary, metric)
        out += [f"## {title}", "", "| 模型 | " + " | ".join(CELLS[HAIKU]) + " |"]
        out.append("|---|" + "---|" * len(CELLS[HAIKU]))
        for m in MODELS:
            out.append(f"| {m} | " + " | ".join(ci.get((m, c), "—") for c in CELLS[HAIKU]) + " |")
        out.append("")
    out.append(f"有答案的題 {len(ans(cell[(OPUS, 'agent')]))}。")

    for title, metric in (("答對", "right"), ("正解列全進", "rows_all_hit")):
        out += ["", f"## 配對差距：{title}（按題 bootstrap，括號是按 cluster）", ""]
        if metric == "right":
            out += ["粗體是跑前宣告的主檢定，只有它們下判定；其餘只描述。", ""]
        out += ["| 配對 | 模型 | 差距 [按題] | [按 cluster] | 判定 |", "|---|---|---|---|---|"]
        out += _diff_lines(diffs, metric)

    for title, metric in (("答對", "right"), ("正解列全進", "rows_all_hit")):
        out += ["", f"## 按鏈：{title}（有答案的題）", ""]
        out += ["| 模型 | 條件 | " + " | ".join(CHAINS) + " |", "|---|---|" + "---|" * len(CHAINS)]
        for m in MODELS:
            for c in CELLS[m]:
                rs = ans(cell[(m, c)])
                if rs and not (metric == "rows_all_hit" and c == "oracle"):
                    out.append(
                        f"| {m} | {c} | "
                        + " | ".join(
                            _mean([r for r in rs if r["chain"] == ch], metric) for ch in CHAINS
                        )
                        + " |"
                    )

    out += ["", "## 正解列全進依據時的答對（有答案的題）", ""]
    out += ["| 模型 | 條件 | 全進：對／題 | 不全：對／題 |", "|---|---|---|---|"]
    for m in MODELS:
        for c in CELLS[m]:
            if c == "oracle":
                continue
            h = by_hit(scored, m, c)
            out.append(
                f"| {m} | {c} | {h['hit'][0]}/{h['hit'][1]} | {h['miss'][0]}/{h['miss'][1]} |"
            )

    out += ["", "## 答出口的（有答案的題：found 為真且有數值）", ""]
    out += ["| 模型 | 條件 | 答了 | 對 | 選小一級 | 其他錯 |", "|---|---|---|---|---|---|"]
    for m in MODELS:
        for c in CELLS[m]:
            a = [r for r in ans(cell[(m, c)]) if r["answered"]]
            right = sum(bool(r["right"]) for r in a)
            crit = sum(bool(r["critical"]) for r in a)
            out.append(f"| {m} | {c} | {len(a)} | {right} | {crit} | {len(a) - right - crit} |")

    searcher = run.make_searcher()
    out += ["", "## 怎麼查的（agent，計分的題）", ""]
    out += [
        "| 模型 | 條件 | 平均查幾次 | 次數分布 0／1／2／3／4／5 | 被擋的查詢 | 平均輪數 "
        "| 被停 | 沒查就答（對／答） | 引句在查到的文字裡（答出口的） "
        "| 引句裡的數字都在查到的文字裡（跑完後加的） |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in MODELS:
        for c in RUNS[m]:
            rs = cell[(m, c)]
            n = Counter(r["n_searches"] for r in rs)
            nosearch = [r for r in rs if r["n_searches"] == 0 and r["answered"]]
            given = [r for r in rs if r["answered"]]
            got = {r["qid"]: [searcher.search(x)[0] for x in r["queries"]] for r in given}
            inq = sum(quote_in(r["quote"], got[r["qid"]]) for r in given)
            nums = sum(numbers_in(r["quote"], got[r["qid"]]) for r in given)
            out.append(
                f"| {m} | {c} | {statistics.mean(r['n_searches'] for r in rs):.2f} "
                f"| {'／'.join(str(n.get(i, 0)) for i in range(6))} "
                f"| {sum(r.get('refused_searches', 0) for r in rs)} "
                f"| {statistics.mean(r.get('turns', 0) for r in rs):.2f} "
                f"| {sum(bool(r.get('stopped')) for r in rs)} "
                f"| {sum(bool(r['right']) for r in nosearch)}／{len(nosearch)} "
                f"| {inq}/{len(given)} | {nums}/{len(given)} |"
            )

    out += ["", "## 該棄答的題、選小一級（critical）", ""]
    out += ["| 模型 | 條件 | 該棄答有棄答 | 選小一級（選型題） |", "|---|---|---|---|"]
    for m in MODELS:
        for c in CELLS[m]:
            ab = [r for r in cell[(m, c)] if r["kind"] == "abstain"]
            ld = [r for r in cell[(m, c)] if r["kind"] == "ladder"]
            out.append(
                f"| {m} | {c} | {_mean(ab, 'abstain_ok')}（{len(ab)}） "
                f"| {_mean(ld, 'critical')}（{len(ld)}） |"
            )

    out += ["", "## 雜訊：Haiku 兩次", ""]
    a1 = {r["qid"]: r for r in cell[(HAIKU, "agent")]}
    a2 = {r["qid"]: r for r in cell[(HAIKU, "agent-2")]}
    common = sorted(set(a1) & set(a2))
    same = sum(same_answer(a1[q]["content"], a2[q]["content"]) for q in common)
    sameq = sum(a1[q]["queries"] == a2[q]["queries"] for q in common)
    ok1 = sum(bool(a1[q]["right"]) for q in common)
    ok2 = sum(bool(a2[q]["right"]) for q in common)
    out += [
        "| 同答 | 查詢逐字相同 | 答對 第一次 → 第二次 |",
        "|---|---|---|",
        f"| {same}/{len(common)} | {sameq}/{len(common)} | {ok1} → {ok2} |",
    ]

    out += ["", "## 花費（每格；含失敗的題次）", ""]
    out += [
        "| 模型 | 條件 | 題次 | US$ | 每題 US$ | 平均 API 呼叫 | 秒（中位） |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in MODELS:
        for c in CELLS[m]:
            k = cost(raw if c.startswith("agent") else raw26, m, c)
            out.append(
                f"| {m} | {c} | {k['asked']} | {k['usd']:.2f} | {k['per_question']:.4f} "
                f"| {k['calls_mean']} | {k['sec_median']} |"
            )

    out += ["", "## 查詢的樣子（Opus，每鏈前 2 題）", ""]
    out += ["| 題 | 查詢 | 撈全 | 答對 |", "|---|---|---|---|"]
    for ch in CHAINS:
        rs = [r for r in ans(cell[(OPUS, "agent")]) if r["chain"] == ch][:2]
        for r in rs:
            qs = " ／ ".join(x.replace("|", "／") for x in r["queries"]) or "（沒查）"
            out.append(f"| {r['qid']} | {qs} | {r['rows_all_hit']} | {r['right']} |")

    (RESULTS / "report.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    logger.info("wrote report", extra={"ctx": {"spent_usd": meta["spent_usd"]}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
