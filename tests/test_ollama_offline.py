"""ollama.chat without a server: the reply fields it passes through (a fake /api/chat reply)."""

from rag_lab import ollama


def test_chat_passes_prompt_eval_count_through(monkeypatch):
    reply = {
        "message": {"content": "{}", "thinking": ""},
        "eval_count": 7,
        "prompt_eval_count": 2345,
        "total_duration": 2e9,
        "done_reason": "stop",
    }
    monkeypatch.setattr(ollama, "_post", lambda path, body, timeout: reply)
    out = ollama.chat("q", model="m")
    assert out["prompt_eval_count"] == 2345
    assert out["eval_count"] == 7 and out["duration"] == 2.0
