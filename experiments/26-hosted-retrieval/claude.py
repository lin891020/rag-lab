"""One-turn Claude Messages API call, optionally with page images, returned in rag_lab.ollama.chat's
shape (as 20's claude.py) plus "served_model" and "request_id". Images ride before the text, in
page order.

No temperature (the API rejects it on these models); thinking is adaptive and cannot be turned off
on Opus 5.5, so depth is set by EFFORT for both models (20's). The reply is constrained by
structured outputs to 06's schema. A stop other than end_turn (refusal, max_tokens) raises
StopError (a ValueError) carrying the tokens it was billed for: it is not an answer. The key comes
from rag-lab/.env (load_env, as in 20 and 22); it is never printed or logged.
"""

from __future__ import annotations

import base64
import copy
import logging
import os
import time
from functools import cache
from pathlib import Path
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

ENV = Path(__file__).parents[2] / ".env"
EFFORT = "medium"  # Opus 5.5's default, written out; the same for Haiku 5.5 (20's)
MAX_TOKENS = 16000
# US$ per million tokens (in, out), from the claude-api skill's models.md on 2026-11-04;
# Haiku's price holds for prompts of 100K tokens or fewer
PRICES = {"claude-opus-5-5": (4.00, 20.00), "claude-haiku-5-5": (0.10, 0.50)}
UNSUPPORTED = ("maxItems", "minItems", "minimum", "maximum", "minLength", "maxLength")


class StopError(ValueError):
    """A reply that stopped for another reason than end_turn; it was still paid for."""

    def __init__(self, msg: str, *, tokens_in: int, tokens_out: int, reason: str = ""):
        super().__init__(msg)
        self.tokens_in, self.tokens_out, self.reason = tokens_in, tokens_out, reason


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


def api_schema(schema: dict) -> dict:
    """A copy with additionalProperties false on every object and the constraints structured
    outputs does not take removed."""

    def walk(x):
        if isinstance(x, dict):
            for k in UNSUPPORTED:
                x.pop(k, None)
            if x.get("type") == "object":
                x["additionalProperties"] = False
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        return x

    return walk(copy.deepcopy(schema))


def content(prompt: str, images: list[tuple[bytes, str]]) -> str | list[dict]:
    """The user turn: the prompt alone, or the pages (bytes, media type) in order, then the text."""
    if not images:
        return prompt
    blocks: list[dict] = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": mt, "data": base64.b64encode(data).decode()},
        }
        for data, mt in images
    ]
    return [*blocks, {"type": "text", "text": prompt}]


def chat(
    prompt: str,
    *,
    model: str,
    system: str,
    schema: dict,
    images: list[tuple[bytes, str]] | tuple = (),
) -> dict:
    t = time.monotonic()
    turn: Any = {"role": "user", "content": content(prompt, list(images))}
    r = _client().messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[turn],
        output_config={
            "effort": EFFORT,
            "format": {"type": "json_schema", "schema": api_schema(schema)},
        },
    )
    if r.stop_reason != "end_turn":
        raise StopError(
            f"stop_reason {r.stop_reason} (request {r._request_id})",
            tokens_in=r.usage.input_tokens,
            tokens_out=r.usage.output_tokens,
            reason=str(r.stop_reason),
        )
    return {
        "content": "".join(b.text for b in r.content if b.type == "text"),
        "thinking": "",  # thinking display is omitted by default; its tokens are in eval_count
        "eval_count": r.usage.output_tokens,
        "prompt_eval_count": r.usage.input_tokens,
        "duration": time.monotonic() - t,
        "done_reason": r.stop_reason,
        "served_model": r.model,
        "request_id": r._request_id,
    }
