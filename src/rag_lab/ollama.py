"""Thin client for the host ollama (127.0.0.1:11434). Standard library only.

Every project on this machine shares that one daemon (see ~/Projects/CLAUDE.md), so the
base URL is fixed here and overridable only through OLLAMA_HOST.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

BASE = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


def _post(path: str, body: dict, timeout: float = 300) -> dict:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def is_up() -> bool:
    """Probe: True when the daemon answers /api/version."""
    try:
        with urllib.request.urlopen(BASE + "/api/version", timeout=2):
            return True
    # swallow: probe -- "not running" is a valid answer, the caller skips or warns
    except (urllib.error.URLError, OSError):
        return False


def version() -> str:
    with urllib.request.urlopen(BASE + "/api/version", timeout=5) as resp:
        return json.load(resp)["version"]


def embed(texts: list[str], *, model: str, batch: int = 64) -> list[list[float]]:
    """One vector per input text, in order. Batched so long lists do not time out."""
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        out.extend(
            _post("/api/embed", {"model": model, "input": texts[i : i + batch]})["embeddings"]
        )
    logger.debug("embedded", extra={"ctx": {"model": model, "n": len(texts)}})
    return out


def model_digest(model: str) -> str:
    """'sha256:<64 hex>' of the local model, from /api/tags. Raises KeyError when not pulled."""
    with urllib.request.urlopen(BASE + "/api/tags", timeout=5) as resp:
        models = json.load(resp)["models"]
    want = model if ":" in model else model + ":latest"
    for m in models:
        if m["name"] == want:
            return "sha256:" + m["digest"]
    raise KeyError(f"model {model!r} not in ollama; run `ollama pull {model}`")


def chat(
    prompt: str,
    *,
    model: str,
    system: str = "",
    format: str | dict | None = None,
    think: bool | str = False,
    options: dict | None = None,
    images: list[str] | None = None,
    timeout: float = 300,
) -> dict:
    """One-turn /api/chat. Returns the raw reply, not parsed:
    {"content", "thinking", "eval_count", "duration", "done_reason"} (duration in seconds).

    `format`: None (free text), "json" (JSON mode) or a JSON-schema dict (constrained decoding).
    `think`: False/True, or a level string ("low") for models whose thinking cannot be turned off
    (gpt-oss ignores False and keeps thinking; under a schema that empties the content).
    `images`: base64-encoded images attached to the user turn (vision models).
    """
    user: dict = {"role": "user", "content": prompt}
    if images:
        user["images"] = images
    msgs = ([{"role": "system", "content": system}] if system else []) + [user]
    body: dict = {
        "model": model,
        "messages": msgs,
        "stream": False,
        "think": think,
        "options": {"temperature": 0, **(options or {})},
    }
    if format is not None:
        body["format"] = format
    resp = _post("/api/chat", body, timeout=timeout)
    msg = resp.get("message", {})
    out = {
        "content": msg.get("content", "") or "",
        "thinking": msg.get("thinking", "") or "",
        "eval_count": int(resp.get("eval_count", 0)),
        "duration": float(resp.get("total_duration", 0)) / 1e9,
        "done_reason": resp.get("done_reason", ""),
    }
    logger.debug(
        "chat",
        extra={
            "ctx": {
                "model": model,
                "format": "schema" if isinstance(format, dict) else format,
                "think": think,
                "prompt_len": len(prompt),
                "images": len(images or []),
                "reply_len": len(out["content"]),
                "thinking_len": len(out["thinking"]),
                "eval_count": out["eval_count"],
                "done_reason": out["done_reason"],
            }
        },
    )
    return out


def chat_json(prompt: str, *, model: str, system: str = "", timeout: float = 300) -> dict:
    """One-turn chat in JSON mode with thinking off. Returns the parsed object."""
    return json.loads(
        chat(prompt, model=model, system=system, format="json", timeout=timeout)["content"]
    )
