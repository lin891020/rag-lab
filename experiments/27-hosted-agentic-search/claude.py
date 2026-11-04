"""Claude with one tool: a manual agent loop over the Messages API (26's claude.py, one turn
there). Each turn sends the whole conversation back, the assistant's turn as it came (thinking
blocks included); tool calls are answered in one user message, each result matched to its
tool_use_id. The final reply is constrained by structured outputs to 26's schema.

No temperature (the API rejects it on these models); effort sets the depth (26's). tool_choice
stays auto (Opus 5.5 rejects forced tool use). Top-level automatic prompt caching: every turn
re-reads the conversation so far. Searches past MAX_SEARCHES get an error result instead of a
search; a question still searching after MAX_TURNS turns is stopped. Every turn is billed into
the caller's `usage`, before anything can raise, so a stopped or failed question keeps its bill.
A stop other than end_turn or tool_use (refusal, max_tokens) and the turn cap raise StopError.
The key comes from rag-lab/.env (load_env, as 20, 22, 25, 26); it is never printed or logged.
"""

from __future__ import annotations

import copy
import logging
import os
import time
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

ENV = Path(__file__).parents[2] / ".env"
EFFORT = "medium"  # Opus 5.5's default, written out; the same for Haiku 5.5 (20's)
MAX_TOKENS = 16000  # per turn
MAX_SEARCHES = 5
MAX_TURNS = 8
LIMIT_MSG = "已達查詢上限（5 次），不能再查，請用已經查到的依據回答。"
EMPTY_MSG = "query 是空的，沒有查。"
UNKNOWN_MSG = "沒有這個工具。"
# US$ per million tokens (in, out), from the claude-api skill's models.md on 2026-11-04;
# Haiku's price holds for prompts of 100K tokens or fewer
PRICES = {"claude-opus-5-5": (4.00, 20.00), "claude-haiku-5-5": (0.10, 0.50)}
CACHE_WRITE = 1.25  # x input, 5-minute TTL (prompt-caching.md)
CACHE_READ = {"claude-opus-5-5": 0.05, "claude-haiku-5-5": 0.10}  # x input ($0.20 on Opus 5.5)
UNSUPPORTED = ("maxItems", "minItems", "minimum", "maximum", "minLength", "maxLength")


class StopError(ValueError):
    """A question that stopped for another reason than end_turn; its turns were still paid for
    (they are in the caller's usage). Carries what the loop did up to the stop."""

    def __init__(
        self,
        msg: str,
        *,
        reason: str = "",
        turns: int = 0,
        searches: int = 0,
        refused: int = 0,
        request_ids: list[str] | None = None,
        served_model: str = "",
    ):
        super().__init__(msg)
        self.reason, self.turns, self.searches, self.refused = reason, turns, searches, refused
        self.request_ids, self.served_model = request_ids or [], served_model


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


def zero_usage() -> dict[str, int]:
    return {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0, "calls": 0}


def price(model: str, usage: dict) -> float:
    """US$ for a usage dict (input, cache_write, cache_read, output); missing keys are 0."""
    pin, pout = PRICES[model]
    usd = (
        usage.get("input", 0) * pin
        + usage.get("cache_write", 0) * pin * CACHE_WRITE
        + usage.get("cache_read", 0) * pin * CACHE_READ[model]
        + usage.get("output", 0) * pout
    )
    return round(usd / 1_000_000, 6)


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


def _bill(usage: dict[str, int], u: Any) -> None:
    usage["input"] += u.input_tokens
    usage["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
    usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
    usage["output"] += u.output_tokens
    usage["calls"] += 1


def _error(block_id: str, msg: str) -> dict:
    return {"type": "tool_result", "tool_use_id": block_id, "content": msg, "is_error": True}


def agent(
    prompt: str,
    *,
    model: str,
    system: str,
    schema: dict,
    tool: dict,
    run_tool: Callable[[str], str],
    usage: dict[str, int],
    max_searches: int | None = None,
    max_turns: int | None = None,
) -> dict:
    max_searches = MAX_SEARCHES if max_searches is None else max_searches
    max_turns = MAX_TURNS if max_turns is None else max_turns
    t = time.monotonic()
    messages: list[Any] = [{"role": "user", "content": prompt}]
    searches = refused = 0
    request_ids: list[str] = []
    for turn in range(1, max_turns + 1):
        r = _client().messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=messages,
            tools=[tool],
            cache_control={"type": "ephemeral"},
            output_config={
                "effort": EFFORT,
                "format": {"type": "json_schema", "schema": api_schema(schema)},
            },
        )
        _bill(usage, r.usage)
        request_ids.append(r._request_id)

        def stop(msg: str, reason: str) -> StopError:
            return StopError(
                msg,
                reason=reason,
                turns=turn,
                searches=searches,
                refused=refused,
                request_ids=request_ids,
                served_model=r.model,
            )

        if r.stop_reason == "tool_use" and turn == max_turns:  # its results would go unread
            raise stop(f"still searching after {max_turns} turns", "max_turns")
        if r.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": r.content})
            results = []
            for b in r.content:
                if b.type != "tool_use":
                    continue
                query = str((b.input or {}).get("query", "")).strip()
                if b.name != tool["name"]:
                    results.append(_error(b.id, UNKNOWN_MSG))
                elif searches >= max_searches:
                    refused += 1
                    results.append(_error(b.id, LIMIT_MSG))
                elif not query:
                    results.append(_error(b.id, EMPTY_MSG))
                else:
                    searches += 1
                    results.append(
                        {"type": "tool_result", "tool_use_id": b.id, "content": run_tool(query)}
                    )
            messages.append({"role": "user", "content": results})
            continue
        if r.stop_reason != "end_turn":
            raise stop(f"stop_reason {r.stop_reason} (request {r._request_id})", str(r.stop_reason))
        return {
            "content": "".join(b.text for b in r.content if b.type == "text"),
            "turns": turn,
            "searches": searches,
            "refused": refused,
            "duration": time.monotonic() - t,
            "done_reason": r.stop_reason,
            "served_model": r.model,
            "request_ids": request_ids,
        }
    raise AssertionError("unreachable: the last turn returns or raises")
