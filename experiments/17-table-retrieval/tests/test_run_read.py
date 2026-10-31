"""The rows copied from 16 are what 17 would have asked: same evidence text, same prompt length,
for struct@1500 and oracle on every question (from the embedding cache, no chat model)."""

import json
from pathlib import Path

import pytest

from evidence import Evidence
from prompts import user_prompt
from retrievers import EmbedCache
from run_read import LABELS, RAW16, missing_reused, read_raw, reused_rows, spread
from run_retrieval import load_questions

ROOT = Path(__file__).parents[3]
CACHE = ROOT / "data/01-chunking/emb-cache"
Q = load_questions()


def test_reused_rows_are_renamed():
    rows = reused_rows()
    assert {r["condition"] for r in rows} == {"struct@1500", "oracle"}
    assert len(rows) == 2 * 2 * 160
    assert all(r["reused_from"] == "16" for r in rows)


def test_spread_reaches_every_chain():
    assert {q["chain"] for q in spread(Q, 10)} == {"C1", "C2", "C3", "C4", "C5"}
    assert len(spread(Q, 10)) == 10 and len(spread(Q, 2)) == 2


def _no_embed(texts):
    raise AssertionError(f"cache miss: {len(texts)} texts")


@pytest.mark.skipif(not CACHE.exists(), reason="embedding cache lives on mini")
def test_17_rebuilds_16s_evidence_and_prompt():
    ev = Evidence(
        Q, cells=["struct@1500"], embed=_no_embed, cache=EmbedCache(sorted(CACHE.glob("*.npz"))[0])
    )
    old = {
        (r["condition"], r["qid"]): r
        for r in map(json.loads, RAW16.read_text("utf-8").splitlines())
        if r["model"] == "qwen3:8b" and r["condition"] in ("L1-struct", "L1-oracle")
    }
    byid = {q["id"]: q for q in Q}
    for (cond, qid), r in old.items():
        cell = "struct@1500" if cond == "L1-struct" else "oracle"
        text, _ = ev.for_question(cell, byid[qid])
        assert text == r["evidence_text"], (cell, qid)
        label = LABELS["oracle"] if cell == "oracle" else LABELS["retrieved"]
        prompt = user_prompt(byid[qid]["question"], evidence=text, label=label, tool=False)
        assert len(prompt) == r["prompt_len"], (cell, qid)


def test_missing_reused_completes_a_partial_copy():
    full = missing_reused([], ["qwen3:8b", "gpt-oss:20b"], {q["id"] for q in Q})
    assert len(full) == 640
    partial = full[:3]
    rest = missing_reused(partial, ["qwen3:8b", "gpt-oss:20b"], {q["id"] for q in Q})
    assert len(rest) == 637


def test_read_raw_skips_a_truncated_last_line(tmp_path):
    p = tmp_path / "raw.jsonl"
    p.write_text('{"a": 1}\n{"a": 2')
    assert read_raw(p) == [{"a": 1}]
