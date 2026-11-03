"""The Claude reading route, offline: what is sent (one image block + the prompt, no
temperature), what comes back, and that a stop other than end_turn is an error that still
carries the tokens it was billed for."""

import base64
from types import SimpleNamespace

import pytest

import claude_ocr
from claude_ocr import OCR_PROMPT, StopError, ocr, price


class _Fake:
    def __init__(self, stop="end_turn", text="<table><tr><td>1</td></tr></table>"):
        self.sent: dict = {}
        self.stop, self.text = stop, text
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kw):
        self.sent = kw
        return SimpleNamespace(
            stop_reason=self.stop,
            content=[SimpleNamespace(type="text", text=self.text)],
            usage=SimpleNamespace(input_tokens=1500, output_tokens=900),
            model=kw["model"],
            _request_id="req_x",
        )


def test_ocr_sends_one_image_and_the_prompt(monkeypatch):
    fake = _Fake()
    monkeypatch.setattr(claude_ocr, "_client", lambda: fake)
    r = ocr(b"\xff\xd8jpegbytes")
    msg = fake.sent["messages"][0]["content"]
    assert msg[0]["type"] == "image" and msg[0]["source"]["media_type"] == "image/jpeg"
    assert base64.b64decode(msg[0]["source"]["data"]) == b"\xff\xd8jpegbytes"
    assert msg[1] == {"type": "text", "text": OCR_PROMPT}
    assert "temperature" not in fake.sent and fake.sent["model"] == "claude-opus-5-5"
    assert r["content"].startswith("<table") and (r["tokens_in"], r["tokens_out"]) == (1500, 900)
    assert r["served_model"] == "claude-opus-5-5"


def test_a_stop_other_than_end_turn_raises_with_tokens(monkeypatch):
    monkeypatch.setattr(claude_ocr, "_client", lambda: _Fake(stop="max_tokens"))
    with pytest.raises(StopError) as e:
        ocr(b"x")
    assert (e.value.tokens_in, e.value.tokens_out) == (1500, 900)


def test_prompt_asks_for_the_page_as_it_is():
    assert "<table>" in OCR_PROMPT and "不要補" in OCR_PROMPT


def test_price_per_million_tokens():
    assert price("claude-opus-5-5", 1_000_000, 1_000_000) == 24.0
