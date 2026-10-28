"""Simulated spot check: a local LLM judges every question against its gold text.

    uv run python experiments/01-chunking/judge.py [model ...]

Stand-in for Mike's 5-question spot check while he is away, not a replacement. For each question
the judge sees the question and the gold 條文/表 text and returns: can the gold text answer it,
what the answer is, how natural the phrasing is (1-5), and whether it copies the statute wording.
Writes results/judge-<model>.jsonl and prints a summary.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from rag_lab import ollama
from rag_lab.log import setup_logging
from rag_lab.pdf import extract_text

sys.path.insert(0, str(Path(__file__).parent))
from chunkers import parse_articles  # noqa: E402

logger = logging.getLogger(__name__)
HERE = Path(__file__).parent

SYSTEM = (
    "你是台灣的電機技師，熟悉《用戶用電設備裝置規則》。你在審一份 RAG 評測考卷。"
    "只回 JSON，不要其他文字。"
)
PROMPT = """題目：{question}

標註的正解來源（{label}）：
<<<
{gold}
>>>

請判斷並回 JSON（鍵名固定）：
{{
  "answerable": true 或 false,   // 只靠上面的正解來源，能不能完整回答這題
  "answer": "你從來源讀出的答案，一句話；答不了就寫空字串",
  "natural": 1 到 5,             // 問法像不像工程師在工地或設計時真的會問的話（5 最像）
  "copied_wording": true 或 false, // 題目是不是幾乎照抄來源的句子
  "comment": "一句話，可空"
}}"""


def main(models: list[str]) -> int:
    setup_logging("rag_lab")
    arts = {str(a.num): a.body for a in parse_articles(extract_text(HERE / "corpus/law.pdf"))}
    tabs = {p.stem: p.read_text(encoding="utf-8") for p in (HERE / "corpus/tables").glob("*.txt")}
    qs = [json.loads(ln) for ln in (HERE / "questions.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    for model in models:
        rows = []
        for q in qs:
            src = arts if q["type"] == "article" else tabs
            gold = "\n\n".join(f"【{g}】\n{src[g]}" for g in q["gold"])
            label = "、".join(("第 " + g + " 條") if q["type"] == "article" else g for g in q["gold"])
            try:
                verdict = ollama.chat_json(PROMPT.format(question=q["question"], label=label, gold=gold), model=model, system=SYSTEM)
            except (json.JSONDecodeError, KeyError, OSError):
                logger.exception("judge call failed", extra={"ctx": {"model": model, "qid": q["id"]}})
                verdict = {"answerable": None, "answer": "", "natural": None, "copied_wording": None, "comment": "call failed"}
            rows.append({"qid": q["id"], "type": q["type"], "model": model, **verdict, "why": q["why"]})
        out = HERE / "results" / f"judge-{model.replace(':', '-')}.jsonl"
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        ok = [r for r in rows if r["answerable"] is not None]
        logger.info(
            "judge done",
            extra={"ctx": {
                "model": model, "n": len(rows), "failed": len(rows) - len(ok),
                "answerable": sum(bool(r["answerable"]) for r in ok),
                "copied": sum(bool(r["copied_wording"]) for r in ok),
                "natural_mean": round(sum(float(r["natural"] or 0) for r in ok) / max(1, len(ok)), 2),
            }},
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["gpt-oss:20b", "qwen3:8b"]))
