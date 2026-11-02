"""One-turn Claude Messages API call, returned in rag_lab.ollama.chat's shape so 19's code runs
unchanged: {"content", "thinking", "eval_count", "prompt_eval_count", "duration", "done_reason"}
plus "served_model" (the model the reply names) and "request_id".

No temperature (the API rejects it on these models); thinking is adaptive and cannot be turned
off on Opus 5.5, so depth is set by EFFORT for both models. The reply is constrained by
structured outputs; the schema sent is api_schema(schema), the prompt keeps 19's text.
A stop other than end_turn (refusal, max_tokens) raises StopError (a ValueError) carrying the
tokens it was billed for: it is not an answer.
"""

from __future__ import annotations

import copy
import logging
import time
from functools import cache

import anthropic

logger = logging.getLogger(__name__)

EFFORT = "medium"  # Opus 5.5's default, written out; the same for Haiku 5.5
MAX_TOKENS = 16000
UNSUPPORTED = ("maxItems", "minItems", "minimum", "maximum", "minLength", "maxLength")


class StopError(ValueError):
    """A reply that stopped for another reason than end_turn; it was still paid for."""

    def __init__(self, msg: str, *, tokens_in: int, tokens_out: int):
        super().__init__(msg)
        self.tokens_in, self.tokens_out = tokens_in, tokens_out


@cache
def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(max_retries=5)  # key from ANTHROPIC_API_KEY


def api_schema(schema: dict) -> dict:
    """A copy with additionalProperties false on every object and the constraints structured
    outputs does not take removed (the caller enforces them, e.g. parse_lookups' cap)."""

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


def chat(prompt: str, *, model: str, system: str, schema: dict) -> dict:
    t = time.monotonic()
    r = _client().messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": prompt}],
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
