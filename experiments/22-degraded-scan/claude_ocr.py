"""One page image -> Claude's reading of its table, as HTML (then 06's html_to_markdown).

The key comes from rag-lab/.env (load_env, as in 20); it is never printed or logged. No
temperature (the API rejects it); thinking is adaptive, depth set by EFFORT, as in 20.
A stop other than end_turn raises StopError (a ValueError) carrying the tokens it was billed for.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from functools import cache
from pathlib import Path

import anthropic

logger = logging.getLogger(__name__)

ENV = Path(__file__).parents[2] / ".env"
OCR_MODEL = "claude-opus-5-5"
EFFORT = "medium"  # 20's
MAX_TOKENS = 16000
PRICES = {"claude-opus-5-5": (4.00, 20.00)}  # US$ / M tokens, as in 20
# the same ask as glm-ocr's "Table Recognition:", written out: HTML with spans, as on the page
OCR_PROMPT = (
    "把這張圖裡的表格轉成 HTML <table>，合併格用 rowspan／colspan。"
    "表號標題放在表格前一行，「註」放在表格後。"
    "只照圖上看得到的字，看不清楚的不要補、也不要更正。只回 HTML，不要其他文字。"
)


class StopError(ValueError):
    """A reply that stopped for another reason than end_turn; it was still paid for."""

    def __init__(self, msg: str, *, tokens_in: int, tokens_out: int):
        super().__init__(msg)
        self.tokens_in, self.tokens_out = tokens_in, tokens_out


def load_env(path: Path = ENV) -> None:
    """KEY=value lines into os.environ (existing variables win); the key is never logged."""
    if not path.exists():
        return
    for ln in path.read_text("utf-8").splitlines():
        k, sep, v = ln.strip().partition("=")
        if sep and k and not k.startswith("#"):
            os.environ.setdefault(k, v.strip().strip("'\""))


@cache
def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(max_retries=5)  # key from ANTHROPIC_API_KEY


def price(model: str, tokens_in: int, tokens_out: int) -> float:
    pin, pout = PRICES[model]
    return round((tokens_in * pin + tokens_out * pout) / 1_000_000, 6)


def ocr(image: bytes, *, media_type: str = "image/jpeg", model: str = OCR_MODEL) -> dict:
    t = time.monotonic()
    r = _client().messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64.b64encode(image).decode(),
                        },
                    },
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }
        ],
        output_config={"effort": EFFORT},
    )
    if r.stop_reason != "end_turn":
        raise StopError(
            f"stop_reason {r.stop_reason} (request {r._request_id})",
            tokens_in=r.usage.input_tokens,
            tokens_out=r.usage.output_tokens,
        )
    return {
        "content": "".join(b.text for b in r.content if b.type == "text"),
        "tokens_in": r.usage.input_tokens,
        "tokens_out": r.usage.output_tokens,
        "duration": time.monotonic() - t,
        "served_model": r.model,
        "request_id": r._request_id,
    }
