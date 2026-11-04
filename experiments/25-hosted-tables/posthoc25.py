"""After score.py: noise, cost and where each way of giving the table wins or loses -> posthoc.md.

    uv run python experiments/25-hosted-tables/posthoc25.py

- cost of every cell from raw.jsonl (every call, failed ones too); accuracy from scored.jsonl
- noise: each cell's two runs, the same answer (value) and the same raw reply
- 表四○／表九二 (glm-ocr returned them as plain text) and 表九三～一 (its OCR text has no title)
- correct per table, the flips against clean, and 24's local models beside them (quoted, not rerun)
"""

from __future__ import annotations

import csv
import json
import logging
import statistics
import sys
from pathlib import Path

from rag_lab.log import setup_logging

HERE = Path(__file__).parent
RESULTS = HERE / "results"
SUMMARY24 = HERE.parent / "24-ocr-header/results/summary.csv"
sys.path.insert(0, str(HERE))
import claude  # noqa: E402
from run import BASE, MODELS, load_questions, read_raw  # noqa: E402
from score import TOL, _parse, _to_float  # noqa: E402

logger = logging.getLogger(__name__)

PLAIN = ("表四○", "表九二")  # glm-ocr returned these as plain text (06, 23)
NO_TITLE = ("表九三～一",)  # 24: without its title the models computed from the note


def answer(content: str) -> float | None:
    obj, ok, _ = _parse(content or "")
    return _to_float(obj.get("value")) if ok and isinstance(obj, dict) else None


def _by(rows: list[dict], model: str, cond: str) -> dict[str, dict]:
    return {r["qid"]: r for r in rows if r["model"] == model and r["condition"] == cond}


def noise(rows: list[dict], model: str, cond: str, qids: list[str]) -> dict[str, int]:
    """A cell's two runs: the same value (two unparseable replies do not count), the same raw
    reply, and how many each got right."""
    a, b = _by(rows, model, cond), _by(rows, model, f"{cond}-2")
    same = 0
    for q in qids:
        x, y = answer(a[q]["content"]), answer(b[q]["content"])
        same += x is not None and y is not None and abs(x - y) <= TOL
    return {
        "same": same,
        "n": len(qids),
        "ok1": sum(bool(a[q]["value_ok"]) for q in qids),
        "ok2": sum(bool(b[q]["value_ok"]) for q in qids),
        "raw_same": sum(a[q]["content"] == b[q]["content"] for q in qids),
    }


def cost(rows: list[dict], model: str, conds: list[str]) -> dict:
    """Every call of these conditions, failed ones too: count, US$, mean tokens, median seconds."""
    rs = [r for r in rows if r["model"] == model and r["condition"] in conds]
    tin = sum(r.get("input_tokens", 0) for r in rs)
    tout = sum(r.get("output_tokens", 0) for r in rs)
    good = [r for r in rs if not r.get("error")]
    usd = round(claude.price(model, tin, tout), 4)
    keys = {(r["condition"], r["qid"]) for r in rs}
    return {
        "calls": len(rs),
        "usd": usd,
        "per_question": round(usd / len(keys), 5) if keys else 0.0,
        "in_mean": round(statistics.mean(r["input_tokens"] for r in good)) if good else 0,
        "out_mean": round(statistics.mean(r["output_tokens"] for r in good)) if good else 0,
        "sec_median": round(statistics.median(r["duration"] for r in good), 1) if good else 0.0,
    }


def _read(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text("utf-8").splitlines() if ln.strip()]


def _reply(r: dict) -> str:
    obj, ok, _ = _parse(r["content"] or "")
    if not (ok and isinstance(obj, dict)):
        return "（無法解析）"
    quote = str(obj.get("quote", "")).replace("|", "／").replace("\n", " ")[:40]
    return f"{obj.get('value')} {obj.get('unit', '')}「{quote}」"


def local24() -> dict[tuple[str, str], str]:
    """(model, condition) -> 24's value_ok mean, for clean and glm2."""
    out = {}
    for r in csv.DictReader(SUMMARY24.open(encoding="utf-8")):
        if r["metric"] == "value_ok" and r["condition"] in ("clean", "glm2"):
            out[(r["model"], r["condition"])] = f"{float(r['mean']):.2f}".lstrip("0")
    return out


def main() -> int:
    setup_logging("rag_lab")
    raw, scored = read_raw(RESULTS / "raw.jsonl"), _read(RESULTS / "scored.jsonl")
    meta = json.loads((RESULTS / "meta.json").read_text("utf-8"))
    questions = load_questions()
    every = [q["id"] for q in questions]
    gold = {q["id"]: q["gold"][0] for q in questions}
    expect = {q["id"]: f"{q['expect']['value']} {q['expect']['unit']}" for q in questions}
    out = ["# 25 posthoc", ""]

    out += [
        f"送出的模型：{', '.join(meta['served_models'])}；anthropic {meta['anthropic']}；"
        f"effort {meta['effort']}；總花費 US${meta['spent_usd']}（含失敗的呼叫）",
        "",
        "## 花費（兩次合計，含失敗的呼叫）",
        "",
        "| 模型 | 條件 | 呼叫 | 被停 | US$ | 每題一次 US$ | 輸入 token（平均） "
        "| 輸出 token（平均） | 秒（中位） |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in MODELS:
        for c in BASE:
            k = cost(raw, m, [c, f"{c}-2"])
            stops = sum(
                bool(r.get("stopped"))
                for r in scored
                if r["model"] == m and r["condition"] in (c, f"{c}-2")
            )
            out.append(
                f"| {m} | {c} | {k['calls']} | {stops} | {k['usd']:.2f} | {k['per_question']:.4f} "
                f"| {k['in_mean']} | {k['out_mean']} | {k['sec_median']} |"
            )

    out += ["", "## 雜訊：同一格的兩次", ""]
    out += [
        "| 模型 | 條件 | 答案相同 | 回覆逐字相同 | 答對 第一次 → 第二次 |",
        "|---|---|---|---|---|",
    ]
    for m in MODELS:
        for c in BASE:
            n = noise(scored, m, c, every)
            out.append(
                f"| {m} | {c} | {n['same']}/{n['n']} | {n['raw_same']}/{n['n']} "
                f"| {n['ok1']} → {n['ok2']} |"
            )

    local = local24()
    out += ["", "## 答對（第一次），並列 24 的本機模型", ""]
    out += ["| 模型 | clean | glm2 | image | image-L2 |", "|---|---|---|---|---|"]
    for m in MODELS:
        cells = {c: _by(scored, m, c) for c in BASE}
        acc = {c: sum(bool(cells[c][q]["value_ok"]) for q in every) / len(every) for c in BASE}
        out.append(f"| {m} | " + " | ".join(f"{acc[c]:.2f}".lstrip("0") for c in BASE) + " |")
    for m in ("qwen3:8b", "gpt-oss:20b"):
        out.append(
            f"| {m}（24） | {local.get((m, 'clean'), '—')} | {local.get((m, 'glm2'), '—')} | — | — |"
        )

    for name, tables in (
        ("回純文字的 表四○、表九二", PLAIN),
        ("OCR 少了標題的 表九三～一", NO_TITLE),
    ):
        qids = [q for q in every if gold[q] in tables]
        out += ["", f"## {name}（{len(qids)} 題，第一次答對）", ""]
        out += ["| 模型 | clean | glm2 | image | image-L2 |", "|---|---|---|---|---|"]
        for m in MODELS:
            cells = {c: _by(scored, m, c) for c in BASE}
            out.append(
                f"| {m} | "
                + " | ".join(str(sum(bool(cells[c][q]["value_ok"]) for q in qids)) for c in BASE)
                + " |"
            )

    out += ["", "## 按表答對（第一次；只列四個條件不全一樣的表）", ""]
    out += [
        "| 模型 | 表 | 題數 | clean | glm2 | image | image-L2 |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in MODELS:
        cells = {c: _by(scored, m, c) for c in BASE}
        for t in sorted(set(gold.values())):
            qids = [q for q in every if gold[q] == t]
            n = [sum(bool(cells[c][q]["value_ok"]) for q in qids) for c in BASE]
            if len(set(n)) > 1:
                out.append(f"| {m} | {t} | {len(qids)} | " + " | ".join(map(str, n)) + " |")

    out += ["", "## 跟 clean 不同對錯的題（第一次）", ""]
    out += [
        "| 模型 | 條件 | 方向 | 題 | 表 | 正解 | clean | 這個條件 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in MODELS:
        ref = _by(scored, m, "clean")
        for c in BASE[1:]:
            cur = _by(scored, m, c)
            for q in every:
                if bool(ref[q]["value_ok"]) != bool(cur[q]["value_ok"]):
                    way = "多對" if cur[q]["value_ok"] else "少對"
                    out.append(
                        f"| {m} | {c} | {way} | {q} | {gold[q]} | {expect[q]} "
                        f"| {_reply(ref[q])} | {_reply(cur[q])} |"
                    )

    (RESULTS / "posthoc.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    logger.info("wrote posthoc", extra={"ctx": {"spent_usd": meta["spent_usd"]}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
