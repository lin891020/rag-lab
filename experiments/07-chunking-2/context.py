"""Contextual retrieval, local edition: qwen3:8b writes one or two sentences of context for
every article chunk (what the article regulates, when it applies), to be prepended before
embedding / BM25. Anthropic's recipe used the whole document as context; here the model gets
the chapter / section path and the article itself (255k characters do not fit num_ctx).

    uv run python experiments/07-chunking-2/context.py   -> corpus/contexts.jsonl (resumable)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from rag_lab import ollama
from rag_lab.log import setup_logging
from rag_lab.pdf import extract_text

HERE = Path(__file__).parent
ONE = HERE.parent / "01-chunking"
OUT = HERE / "corpus/contexts.jsonl"
sys.path.insert(0, str(HERE))
from chunkers import Chunk, article_chunks, parse_articles  # noqa: E402

logger = logging.getLogger(__name__)

MODEL = "qwen3:8b"
OPTIONS = {"temperature": 0, "seed": 0, "num_ctx": 4096, "num_predict": 160}
SYSTEM = "你是台灣的電機技師，熟悉《用戶用電設備裝置規則》。只輸出要求的句子，不要標題、不要條列、不要解釋。"
PROMPT = """下面是《用戶用電設備裝置規則》的一條（或一條的一段）：

<<<
{chunk}
>>>

用一到兩句話（總共不超過 60 字）說明這一條在規範什麼、工程師在什麼情境會查到它。這段話會接在條文前面給搜尋用，所以要寫出條文裡沒直接寫、但問的人會用的講法（設備名稱、常見問法）。直接輸出句子。"""


def chunk_key(c: Chunk, idx: int) -> str:
    return f"{c.meta['article']}:{idx}"


def load_chunks() -> list[Chunk]:
    return article_chunks(parse_articles(extract_text(ONE / "corpus/law.pdf")), max_len=1000)


def load_contexts() -> dict[str, str]:
    if not OUT.exists():
        return {}
    return {
        json.loads(ln)["key"]: json.loads(ln)["context"]
        for ln in OUT.read_text("utf-8").splitlines()
        if ln.strip()
    }


def main() -> int:
    setup_logging("rag_lab")
    chunks = load_chunks()
    done = load_contexts()
    todo = [(chunk_key(c, i), c) for i, c in enumerate(chunks) if chunk_key(c, i) not in done]
    logger.info("contexts", extra={"ctx": {"chunks": len(chunks), "todo": len(todo)}})
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("a", encoding="utf-8") as f:
        for n, (key, c) in enumerate(todo, 1):
            try:
                r = ollama.chat(
                    PROMPT.format(chunk=c.text),
                    model=MODEL,
                    system=SYSTEM,
                    think=False,
                    options=OPTIONS,
                )
                ctx = r["content"].strip().replace("\n", " ")
            except (OSError, KeyError, ValueError):
                logger.exception("context failed", extra={"ctx": {"key": key}})
                ctx = ""
            f.write(
                json.dumps(
                    {"key": key, "labels": c.labels, "context": ctx, "chars": len(ctx)},
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()
            if n % 50 == 0 or n == len(todo):
                logger.info("progress", extra={"ctx": {"done": n, "todo": len(todo)}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
