"""embed() against a fake server: a 413 on a batch is redone one text at a time, and a text that
still overruns the context is halved until it embeds (no real ollama needed)."""

import io
import urllib.error
from email.message import Message

import rag_lab.ollama as ol


def _http413():
    return urllib.error.HTTPError(
        "http://x/api/embed",
        413,
        "Request Entity Too Large",
        Message(),
        io.BytesIO(b'{"error":"the input length exceeds the context length"}'),
    )


def test_batch_413_falls_back_to_single_texts_and_halving(monkeypatch, caplog):
    calls = []

    def fake_post(path, body, timeout=300):
        inputs = body["input"]
        calls.append(list(inputs))
        if len(inputs) > 1 or len(inputs[0]) > 1000:
            raise _http413()
        return {"embeddings": [[float(len(t))] for t in inputs]}

    monkeypatch.setattr(ol, "_post", fake_post)
    with caplog.at_level("WARNING"):
        vecs = ol.embed(["a" * 300, "b" * 4000], model="bge-m3")
    assert vecs == [[300.0], [1000.0]]  # the long one was halved 4000 -> 2000 -> 1000
    assert calls[0] == ["a" * 300, "b" * 4000] and calls[1] == ["a" * 300]
    assert any("embed input cut" in r.message for r in caplog.records)


def test_other_http_errors_still_raise(monkeypatch):
    def fake_post(path, body, timeout=300):
        raise urllib.error.HTTPError("http://x", 500, "boom", Message(), io.BytesIO(b""))

    monkeypatch.setattr(ol, "_post", fake_post)
    try:
        ol.embed(["x"], model="bge-m3")
    except urllib.error.HTTPError as exc:
        assert exc.code == 500
    else:
        raise AssertionError("500 must propagate")
