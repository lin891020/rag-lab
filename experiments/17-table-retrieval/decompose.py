"""qwen3:8b splits each question into 1–3 retrieval queries ("which rule / which table to look
up"), never the answer -> results/subqueries.jsonl (resumable: done ids are skipped). One file for
both reading models.

    uv run python experiments/17-table-retrieval/decompose.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from rag_lab import ollama
from rag_lab.log import setup_logging

HERE = Path(__file__).parent
OUT = HERE / "results" / "subqueries.jsonl"
QUESTIONS = HERE / "questions.jsonl"
MODEL = "qwen3:8b"
OPTIONS = {"temperature": 0, "seed": 0, "num_ctx": 4096, "num_predict": 256}
MAX_QUERIES = 3

logger = logging.getLogger(__name__)

SYSTEM = "你是檢索助手，幫忙把問題拆成搜尋查詢。只回一個 JSON 物件，不要其他文字。"
SCHEMA: dict = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
}
PROMPT = """下面是一個關於屋內線路裝置規則的問題。回答它可能要查好幾條規定或好幾張附表。
請列出 1 到 3 個搜尋查詢，每個查詢描述一份要找的資料（某一條規定，或某一張表的名稱與內容），讓搜尋引擎能分別找到它們。
不要回答問題、不要計算、不要寫出你猜的數值。

問題：{question}

回傳格式：{{"queries": ["查詢一", "查詢二"]}}"""

_NUM = re.compile(r"\d+(?:\.\d+)?")


def parse(content: str) -> list[str]:
    """The queries, stripped, empty and duplicate ones dropped, at most MAX_QUERIES."""
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        return []
    qs = obj.get("queries") if isinstance(obj, dict) else None
    if not isinstance(qs, list):
        return []
    out: list[str] = []
    for s in qs:
        if isinstance(s, str) and s.strip() and s.strip() not in out:
            out.append(s.strip())
    return out[:MAX_QUERIES]


def new_numbers(question: str, queries: list[str]) -> list[str]:
    """Numbers in the queries that the question does not contain (a guessed value leaking in)."""
    have = set(_NUM.findall(question.replace(",", "")))
    return sorted({n for s in queries for n in _NUM.findall(s.replace(",", ""))} - have)


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    rows = [json.loads(ln) for ln in path.read_text("utf-8").splitlines() if ln.strip()]
    return {r["qid"] for r in rows if not r["error"]}  # a failed one is redone


def decompose(q: dict, *, chat=ollama.chat) -> dict:
    row = {"qid": q["id"], "chain": q["chain"]}
    try:
        r = chat(
            PROMPT.format(question=q["question"]),
            model=MODEL,
            system=SYSTEM,
            format=SCHEMA,
            think=False,
            options=OPTIONS,
        )
        queries = parse(r["content"])
        row.update(queries=queries, raw=r["content"], error="" if queries else "unparsed")
    except (OSError, KeyError, ValueError) as exc:
        logger.exception("decompose failed", extra={"ctx": {"qid": q["id"]}})
        row.update(queries=[], raw="", error=repr(exc))
    row["new_numbers"] = new_numbers(q["question"], row["queries"])
    return row


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    setup_logging("rag_lab")
    questions = [json.loads(ln) for ln in QUESTIONS.read_text("utf-8").splitlines() if ln.strip()]
    if args.limit:
        questions = questions[: args.limit]
    done = done_ids(args.out)  # a redone id appends a second row; readers keep the last one
    todo = [q for q in questions if q["id"] not in done]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    bad = 0
    with args.out.open("a", encoding="utf-8") as f:
        for q in todo:
            row = decompose(q)
            bad += bool(row["error"])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
    logger.info(
        "decomposed", extra={"ctx": {"new": len(todo), "failed": bad, "out": str(args.out)}}
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
