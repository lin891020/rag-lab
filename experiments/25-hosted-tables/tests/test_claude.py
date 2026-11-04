"""The Claude call: page images ride before the text in page order, the schema closes every object,
a stop other than end_turn carries the tokens it was billed for, and the reply comes back in
ollama.chat's shape."""

import os
from types import SimpleNamespace

import claude
import pytest


def test_content_puts_images_first_in_page_order():
    blocks = claude.content("Q", [(b"a", "image/png"), (b"b", "image/jpeg")])
    assert isinstance(blocks, list)
    assert [b["type"] for b in blocks] == ["image", "image", "text"]
    assert [b["source"]["media_type"] for b in blocks[:2]] == ["image/png", "image/jpeg"]
    assert blocks[0]["source"]["data"] == "YQ==" and blocks[2]["text"] == "Q"
    assert claude.content("Q", []) == "Q"


def test_api_schema_closes_objects_and_drops_unsupported_keys():
    s = claude.api_schema({"type": "object", "properties": {"v": {"type": "number", "minimum": 0}}})
    assert s["additionalProperties"] is False and "minimum" not in s["properties"]["v"]


def test_price_per_million_tokens():
    assert claude.price("claude-opus-5-5", 1_000_000, 1_000_000) == 24.0
    assert claude.price("claude-haiku-5-5", 1_000_000, 1_000_000) == 0.60


class _Fake:
    def __init__(self, stop="end_turn"):
        self.stop, self.sent = stop, []
        self.messages = self

    def create(self, **kw):
        self.sent.append(kw)
        return SimpleNamespace(
            stop_reason=self.stop,
            content=[SimpleNamespace(type="text", text='{"quote": "", "value": 1, "unit": ""}')],
            usage=SimpleNamespace(input_tokens=100, output_tokens=7),
            model=kw["model"],
            _request_id="req_1",
        )


def test_chat_sends_images_and_returns_ollama_shape(monkeypatch):
    fake = _Fake()
    monkeypatch.setattr(claude, "_client", lambda: fake)
    r = claude.chat(
        "Q",
        model="claude-haiku-5-5",
        system="S",
        schema={"type": "object", "properties": {}},
        images=[(b"a", "image/png")],
    )
    sent = fake.sent[0]
    assert sent["messages"][0]["content"][0]["type"] == "image" and sent["system"] == "S"
    assert sent["output_config"]["effort"] == claude.EFFORT and "temperature" not in sent
    assert (r["prompt_eval_count"], r["eval_count"], r["done_reason"]) == (100, 7, "end_turn")
    assert r["served_model"] == "claude-haiku-5-5" and r["request_id"] == "req_1"


def test_a_stop_other_than_end_turn_carries_its_tokens(monkeypatch):
    monkeypatch.setattr(claude, "_client", lambda: _Fake(stop="refusal"))
    with pytest.raises(claude.StopError) as e:
        claude.chat("Q", model="claude-opus-5-5", system="S", schema={"type": "object"})
    assert (e.value.tokens_in, e.value.tokens_out, e.value.reason) == (100, 7, "refusal")


def test_load_env_keeps_existing_variables_and_skips_comments(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("# c\nA_KEY='x'\nB_KEY=y\n")
    monkeypatch.setenv("B_KEY", "kept")
    monkeypatch.delenv("A_KEY", raising=False)
    claude.load_env(p)
    assert os.environ["A_KEY"] == "x" and os.environ["B_KEY"] == "kept"
