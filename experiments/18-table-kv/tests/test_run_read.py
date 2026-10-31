"""The rows copied from 17 are what 18 would have asked (qwen3 cards: same evidence text, same
prompt length, from the embedding cache, no chat model); every row records the prompt tokens."""

from pathlib import Path

import pytest

import run_read
from evidence import Evidence
from prompts import user_prompt
from retrievers import EmbedCache
from run_read import LABEL, RAW17, ask, read_raw, rows17, reused_rows
from run_retrieval import load_questions

ROOT = Path(__file__).parents[3]
CACHE = ROOT / "data/01-chunking/emb-cache"
Q = load_questions()


def test_reused_rows_are_qwen3_cards_from_17():
    rows = reused_rows()
    assert len(rows) == 160
    assert {(r["model"], r["condition"], r["reused_from"]) for r in rows} == {
        ("qwen3:8b", "cards", "17")
    }
    assert len({r["qid"] for r in rows}) == 160


def test_rows17_keeps_the_last_good_row_per_key():
    rows = rows17(RAW17)
    struct = [r for r in rows if r["model"] == "gpt-oss:20b" and r["condition"] == "struct@1500"]
    assert len(struct) == 160 and not any(r.get("reused_from") for r in struct)  # 17's redo


def _no_embed(texts):
    raise AssertionError(f"cache miss: {len(texts)} texts")


@pytest.mark.skipif(not CACHE.exists(), reason="embedding cache lives on mini")
def test_18_rebuilds_17s_cards_evidence_and_prompt():
    ev = Evidence(
        Q, cells=["cards"], embed=_no_embed, cache=EmbedCache(sorted(CACHE.glob("*.npz"))[0])
    )
    byid = {q["id"]: q for q in Q}
    for r in reused_rows():
        text, _ = ev.for_question("cards", byid[r["qid"]])
        assert text == r["evidence_text"], r["qid"]
        prompt = user_prompt(byid[r["qid"]]["question"], evidence=text, label=LABEL, tool=False)
        assert len(prompt) == r["prompt_len"], r["qid"]


class _Ev:
    def for_question(self, cell, q):
        return "【段 1】\n表", {"rows_hit": 1, "rows_total": 1}


def test_ask_records_prompt_tokens(monkeypatch):
    reply = {
        "content": "{}",
        "thinking": "",
        "eval_count": 3,
        "prompt_eval_count": 1234,
        "duration": 1.0,
        "done_reason": "stop",
    }
    monkeypatch.setattr(run_read.ollama, "chat", lambda *a, **k: reply)
    row = ask("qwen3:8b", "cards-kv", Q[0], _Ev())
    assert row["prompt_eval_count"] == 1234 and row["error"] == ""


def test_read_raw_skips_a_truncated_last_line(tmp_path):
    p = tmp_path / "raw.jsonl"
    p.write_text('{"a": 1}\n{"a": 2')
    assert read_raw(p) == [{"a": 1}]
