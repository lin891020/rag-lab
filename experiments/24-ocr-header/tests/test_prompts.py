"""Every condition sends the same prompt; only the evidence block (the table's rendering) differs.
clean and glm2 must be the very prompts 23 sent (its clean and glm2-s100): that makes 23's rows a
noise floor for this run."""

import json
from pathlib import Path

from prompts import CONDITIONS, SCHEMA, SYSTEM, user_prompt


def test_conditions():
    assert list(CONDITIONS) == ["clean", "glm2", "glm2-flat"]
    assert all(v in (None, k) for k, v in CONDITIONS.items())  # condition name = corpus route
    assert list(SCHEMA["properties"]) == ["quote", "value", "unit"]
    assert SYSTEM


def test_prompt_differs_only_in_evidence():
    a = user_prompt("幾安培？", evidence="| a | b |", label="表一")
    b = user_prompt("幾安培？", evidence="<table>", label="表一")
    assert a.split("依據")[0] == b.split("依據")[0]
    assert a.split("問題：")[1] == b.split("問題：")[1]
    assert json.dumps(SCHEMA, ensure_ascii=False) in a and "| a | b |" in a


def test_closed_book_says_so():
    assert "不提供" in user_prompt("幾安培？", evidence=None, label="")


def _prompts(cond, questions, corpus):
    from run import evidence_for

    out = {}
    for q in questions:
        text, label = evidence_for(q, cond, corpus)
        out[q["id"]] = user_prompt(q["question"], evidence=text, label=label)
    return out


def test_clean_and_glm2_are_the_prompts_23_sent():
    from run import last_rows, load_corpus, load_questions

    raw23 = Path(__file__).parents[2] / "23-ocr-size/results/raw.jsonl"
    sent = {(r["condition"], r["qid"]): r["prompt_len"] for r in last_rows(raw23)}
    qs, corpus = load_questions(), load_corpus()
    for cond, cond23 in (("clean", "clean"), ("glm2", "glm2-s100")):
        for qid, prompt in _prompts(cond, qs, corpus).items():
            assert len(prompt) == sent[(cond23, qid)], (cond, qid)


def test_flat_changes_the_prompt_of_the_42_multi_header_questions_only():
    from run import load_corpus, load_questions

    qs, corpus = load_questions(), load_corpus()
    a, b = _prompts("glm2", qs, corpus), _prompts("glm2-flat", qs, corpus)
    assert sum(a[q] != b[q] for q in a) == 42
