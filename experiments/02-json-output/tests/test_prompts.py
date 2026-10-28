"""The three open-book conditions must send byte-identical prompts; only `format` may differ."""

import json

from prompts import CONDITIONS, SCHEMA, SYSTEM, user_prompt


def test_four_conditions_with_the_expected_format_field():
    assert set(CONDITIONS) == {"closed-book", "prompt-only", "json-mode", "schema"}
    assert CONDITIONS["prompt-only"]["format"] is None
    assert CONDITIONS["json-mode"]["format"] == "json"
    assert CONDITIONS["schema"]["format"] == SCHEMA
    assert CONDITIONS["closed-book"]["evidence"] is False
    assert all(CONDITIONS[c]["evidence"] for c in ("prompt-only", "json-mode", "schema"))


def test_schema_has_quote_before_value():
    keys = list(SCHEMA["properties"])
    assert keys == ["quote", "value", "unit"]
    assert SCHEMA["required"] == keys


def test_user_prompt_contains_schema_text_evidence_and_question():
    p = user_prompt("幾安培？", evidence="【第 19 條】\n一‧二五倍", label="第 19 條")
    assert json.dumps(SCHEMA, ensure_ascii=False) in p
    assert "一‧二五倍" in p and "幾安培？" in p and "第 19 條" in p


def test_closed_book_prompt_says_no_evidence_and_is_otherwise_the_same():
    a = user_prompt("幾安培？", evidence=None, label="")
    b = user_prompt("幾安培？", evidence="X", label="L")
    assert "不提供" in a and "X" not in a
    assert a.split("問題：")[1] == b.split("問題：")[1]
    assert SYSTEM
