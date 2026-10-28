"""Talks to the real host ollama (127.0.0.1:11434); skipped when it is not up."""

import pytest

from rag_lab.ollama import embed, model_digest

pytestmark = pytest.mark.skipif(
    __import__("rag_lab.ollama", fromlist=["is_up"]).is_up() is False, reason="ollama not running"
)


def test_embed_returns_one_1024_vector_per_input():
    vecs = embed(["分路導線最小線徑", "接地電阻"], model="bge-m3")
    assert len(vecs) == 2
    assert len(vecs[0]) == 1024
    assert vecs[0] != vecs[1]


def test_embed_empty_input_returns_empty_list():
    assert embed([], model="bge-m3") == []


def test_model_digest_is_sha256_hex():
    d = model_digest("bge-m3")
    assert d.startswith("sha256:") and len(d) == len("sha256:") + 64


def test_chat_json_returns_parsed_object():
    from rag_lab.ollama import chat_json

    out = chat_json('回答 JSON：{"n": 2 加 3 的結果}', model="qwen3:8b")
    assert isinstance(out, dict) and int(out.get("n", -1)) == 5


SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "number"}, "unit": {"type": "string"}},
    "required": ["value", "unit"],
}
Q = "2.5 加 1 等於多少？只回 JSON，鍵 value（數字，照算出的值，不要四捨五入）與 unit（字串，填「個」）。"


def test_chat_returns_raw_fields_without_parsing():
    import json

    from rag_lab.ollama import chat

    out = chat(Q, model="qwen3:8b", format=SCHEMA)
    assert set(out) >= {"content", "thinking", "eval_count", "duration", "done_reason"}
    assert isinstance(out["content"], str) and json.loads(out["content"])["value"] == 3.5
    assert out["eval_count"] > 0 and out["duration"] > 0


def test_chat_schema_with_gpt_oss_needs_think_low():
    """gpt-oss ignores think=False; with think='low' the schema answer is non-empty."""
    import json

    from rag_lab.ollama import chat

    out = chat(Q, model="gpt-oss:20b", format=SCHEMA, think="low")
    assert json.loads(out["content"])["value"] == 3.5
    assert len(out["thinking"]) > 0


def test_chat_json_still_parses():
    from rag_lab.ollama import chat_json

    assert chat_json(Q, model="qwen3:8b")["value"] == 3.5
