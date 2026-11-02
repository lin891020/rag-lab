"""claude.chat: the Messages API reply in rag_lab.ollama.chat's shape, the schema made acceptable
to structured outputs (additionalProperties false on every object, no maxItems), and any stop
other than end_turn raised (a refusal or a cut-off reply is not an answer)."""

from types import SimpleNamespace

import pytest

import claude
from prompts import LOOKUP_SCHEMA, SCHEMA


def _reply(text, stop="end_turn"):
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        stop_reason=stop,
        model="claude-test",
        usage=SimpleNamespace(input_tokens=100, output_tokens=7),
        _request_id="req_1",
    )


class _Fake:
    def __init__(self, reply):
        self.reply, self.kw = reply, None
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.kw = kw
        return self.reply


def test_api_schema_closes_objects_and_drops_max_items():
    s = claude.api_schema(LOOKUP_SCHEMA)
    assert s["additionalProperties"] is False
    arr = s["properties"]["lookups"]
    assert "maxItems" not in arr and arr["items"]["additionalProperties"] is False
    assert "maxItems" in LOOKUP_SCHEMA["properties"]["lookups"]  # the prompt's copy is untouched
    assert claude.api_schema(SCHEMA)["additionalProperties"] is False


def test_chat_returns_ollama_shape(monkeypatch):
    fake = _Fake(_reply('{"found": false}'))
    monkeypatch.setattr(claude, "_client", lambda: fake)
    r = claude.chat("Q", model="claude-test", system="S", schema=SCHEMA)
    assert r["content"] == '{"found": false}' and r["thinking"] == ""
    assert (r["prompt_eval_count"], r["eval_count"], r["done_reason"]) == (100, 7, "end_turn")
    assert r["served_model"] == "claude-test"
    kw = fake.kw
    assert kw is not None and "temperature" not in kw
    assert kw["output_config"]["effort"] == claude.EFFORT
    assert kw["output_config"]["format"]["schema"]["additionalProperties"] is False


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_chat_raises_on_other_stops(monkeypatch, stop):
    monkeypatch.setattr(claude, "_client", lambda: _Fake(_reply("{", stop)))
    with pytest.raises(claude.StopError, match=stop) as err:
        claude.chat("Q", model="m", system="S", schema=SCHEMA)
    assert (err.value.tokens_in, err.value.tokens_out) == (100, 7)
