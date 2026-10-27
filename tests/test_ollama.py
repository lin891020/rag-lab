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
