"""The evidence builder is 10's (which was 09's): rebuild 09's `all` condition on 09's bank and
match the evidence sizes 09 recorded (needs 09's embedding cache; no ollama call)."""

import json

import pytest

from run_read import CORPUS09, DATA, DEFAULT_CONDITIONS, Evidence, load_docs, load_questions

RAW09 = CORPUS09.parent / "results" / "read" / "raw.jsonl"


@pytest.mark.skipif(
    not RAW09.exists() or not list((DATA / "emb-cache").glob("*.npz")),
    reason="09 results or embedding cache missing on this machine",
)
def test_evidence_matches_what_09_fed_the_models():
    docs, questions = load_docs(), load_questions(CORPUS09.parent / "questions.jsonl")
    ev = Evidence(docs, questions, conditions=["-ocr", "oracle"], embed=_no_ollama)
    want = {
        (r["qid"], r["condition"]): r["evidence_chars"]
        for r in map(json.loads, RAW09.read_text("utf-8").splitlines())
        if r["model"] == "qwen3:8b" and r["condition"] in ("all", "oracle")
    }
    for q in questions[:: len(questions) // 12]:
        text, _, _ = ev.for_question("-ocr", q)
        assert len(text or "") == want[(q["id"], "all")], q["id"]
        text, _, _ = ev.for_question("oracle", q)
        assert len(text or "") == want[(q["id"], "oracle")], q["id"]


def test_default_conditions_are_the_pre_registered_eight():
    assert DEFAULT_CONDITIONS == [
        "closed-book",
        "oracle",
        "fixed",
        "+cnpre",
        "+rows",
        "full",
        "-cnpre",
        "-rows",
    ]


def _no_ollama(texts):
    if texts:
        raise AssertionError(f"embedding cache miss for {len(texts)} texts")
    return []
