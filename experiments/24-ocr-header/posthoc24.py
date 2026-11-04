"""After score.py: what flattening changed, question by question -> results/posthoc.md.

    uv run python experiments/24-ocr-header/posthoc24.py

- noise floor: clean and glm2 send the very prompts 23 sent (its clean and glm2-s100)
- negative control: the 40 questions whose prompt flattening did not change
- the same answer (value) as clean: multi-row-header questions vs the rest, glm2 vs glm2-flat
- correct per table, the flips glm2 -> glm2-flat, and what clean still gets that glm2-flat misses
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from rag_lab.log import setup_logging

HERE = Path(__file__).parent
RESULTS = HERE / "results"
SCORED23 = HERE.parent / "23-ocr-size/results/scored.jsonl"
sys.path.insert(0, str(HERE))
from run import MODELS, load_corpus, load_questions  # noqa: E402
from score import TOL, _parse, _to_float  # noqa: E402

logger = logging.getLogger(__name__)

SAME_PROMPT_AS_23 = {"clean": "clean", "glm2": "glm2-s100"}


def answer(content: str) -> float | None:
    obj, ok, _ = _parse(content or "")
    return _to_float(obj.get("value")) if ok and isinstance(obj, dict) else None


def _by(rows: list[dict], model: str, cond: str) -> dict[str, dict]:
    return {r["qid"]: r for r in rows if r["model"] == model and r["condition"] == cond}


def _same(x: float | None, y: float | None) -> bool:
    """Two unparseable replies are not the same answer."""
    return x is not None and y is not None and abs(x - y) <= TOL


def same_answers(
    rows_x: list[dict], rows_y: list[dict], model: str, cx: str, cy: str, qids: list[str]
) -> int:
    x, y = _by(rows_x, model, cx), _by(rows_y, model, cy)
    return sum(_same(answer(x[q]["content"]), answer(y[q]["content"])) for q in qids)


def same_replies(
    rows_x: list[dict], rows_y: list[dict], model: str, cx: str, cy: str, qids: list[str]
) -> int:
    x, y = _by(rows_x, model, cx), _by(rows_y, model, cy)
    return sum(x[q]["content"] == y[q]["content"] for q in qids)


def flips(rows: list[dict], model: str, before: str, after: str) -> dict[str, list[str]]:
    b, a = _by(rows, model, before), _by(rows, model, after)
    qids = sorted(b)
    return {
        "won": [q for q in qids if not b[q]["value_ok"] and a[q]["value_ok"]],
        "lost": [q for q in qids if b[q]["value_ok"] and not a[q]["value_ok"]],
    }


def changed_questions(
    questions: list[dict], corpus: dict[str, dict[str, str]], a: str, b: str
) -> set[str]:
    return {q["id"] for q in questions if any(corpus[a][g] != corpus[b][g] for g in q["gold"])}


def _read(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text("utf-8").splitlines() if ln.strip()]


def _reply(r: dict) -> str:
    obj, ok, _ = _parse(r["content"] or "")
    if not (ok and isinstance(obj, dict)):
        return "（無法解析）"
    quote = str(obj.get("quote", "")).replace("|", "／")[:40]
    return f"{obj.get('value')}「{quote}」"


def main() -> int:
    setup_logging("rag_lab")
    rows, rows23 = _read(RESULTS / "scored.jsonl"), _read(SCORED23)
    questions = load_questions()
    gold = {q["id"]: q["gold"][0] for q in questions}
    expect = {q["id"]: q["expect"]["value"] for q in questions}
    multi = changed_questions(questions, load_corpus(), "glm2", "glm2-flat")
    every = [q["id"] for q in questions]
    groups = {"多層表頭": sorted(multi), "其他": [q for q in every if q not in multi]}
    out = ["# 24 posthoc", ""]

    meta, meta23 = (
        json.loads((d / "meta.json").read_text("utf-8")) for d in (RESULTS, SCORED23.parent)
    )
    out += ["## 版本：這次與 23", "", "| 項目 | 這次 | 23 | |", "|---|---|---|---|"]
    pairs = [
        ("機器", meta["machine"], meta23["machine"]),
        ("ollama", meta["ollama"], meta23["ollama"]),
    ]
    pairs += [(m, meta["models"][m], meta23["models"].get(m, "")) for m in meta["models"]]
    for name, now, then in pairs:
        out.append(f"| {name} | {now} | {then} | {'相同' if now == then else '**不同**'} |")
    out.append("")

    out += ["## 雜訊底線：同一份提示，23 問過一次", ""]
    out += [
        "| 模型 | 條件（23 的） | 回覆逐字相同 | 答案相同 | 答對：這次 − 23 |",
        "|---|---|---|---|---|",
    ]
    for m in MODELS:
        for c, c23 in SAME_PROMPT_AS_23.items():
            now, then = _by(rows, m, c), _by(rows23, m, c23)
            d = sum(now[q]["value_ok"] for q in every) - sum(then[q]["value_ok"] for q in every)
            out.append(
                f"| {m} | {c}（{c23}） | {same_replies(rows, rows23, m, c, c23, every)}/{len(every)} "
                f"| {same_answers(rows, rows23, m, c, c23, every)}/{len(every)} | {d:+d} 題 |"
            )

    other = groups["其他"]
    out += ["", f"## 陰性對照：攤平沒改到提示的 {len(other)} 題（glm2 對 glm2-flat）", ""]
    out += ["| 模型 | 回覆逐字相同 | 答案相同 |", "|---|---|---|"]
    for m in MODELS:
        out.append(
            f"| {m} | {same_replies(rows, rows, m, 'glm2', 'glm2-flat', other)}/{len(other)} "
            f"| {same_answers(rows, rows, m, 'glm2', 'glm2-flat', other)}/{len(other)} |"
        )

    out += ["", "## 跟 clean 同答的題數（答案＝value）", ""]
    out += ["| 模型 | 題組 | 題數 | glm2 | glm2-flat |", "|---|---|---|---|---|"]
    for m in MODELS:
        for name, qids in groups.items():
            out.append(
                f"| {m} | {name} | {len(qids)} | {same_answers(rows, rows, m, 'glm2', 'clean', qids)} "
                f"| {same_answers(rows, rows, m, 'glm2-flat', 'clean', qids)} |"
            )

    tables = sorted({gold[q] for q in multi})
    out += ["", "## 按表答對（多層表頭的表）", ""]
    out += ["| 模型 | 表 | 題數 | clean | glm2 | glm2-flat |", "|---|---|---|---|---|---|"]
    for m in MODELS:
        cells = {c: _by(rows, m, c) for c in ("clean", "glm2", "glm2-flat")}
        for t in tables:
            qids = [q for q in every if gold[q] == t]
            n = {c: sum(cells[c][q]["value_ok"] for q in qids) for c in cells}
            out.append(
                f"| {m} | {t} | {len(qids)} | {n['clean']} | {n['glm2']} | {n['glm2-flat']} |"
            )

    out += ["", "## 翻面：glm2 → glm2-flat", ""]
    out += ["| 模型 | 方向 | 題 | 表 | 正解 | glm2 | glm2-flat |", "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        g, f = _by(rows, m, "glm2"), _by(rows, m, "glm2-flat")
        for way, qids in flips(rows, m, "glm2", "glm2-flat").items():
            for q in qids:
                out.append(
                    f"| {m} | {'救回' if way == 'won' else '弄錯'} | {q} | {gold[q]} | {expect[q]} "
                    f"| {_reply(g[q])} | {_reply(f[q])} |"
                )

    out += ["", "## clean 答對、glm2-flat 答錯", ""]
    out += ["| 模型 | 題 | 表 | 正解 | glm2-flat |", "|---|---|---|---|---|"]
    for m in MODELS:
        f = _by(rows, m, "glm2-flat")
        for q in flips(rows, m, "clean", "glm2-flat")["lost"]:
            out.append(f"| {m} | {q} | {gold[q]} | {expect[q]} | {_reply(f[q])} |")

    (RESULTS / "posthoc.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    logger.info("wrote posthoc", extra={"ctx": {"multi": len(multi), "other": len(other)}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
