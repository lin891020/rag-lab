"""Every condition sends the same prompt; only the evidence block (the table's rendering) differs."""

import json

from prompts import CONDITIONS, SCHEMA, SYSTEM, user_prompt


def test_conditions():
    assert list(CONDITIONS) == ["closed-book", "clean", "glm", "glm2", "vision", "tesseract"]
    assert CONDITIONS["closed-book"] is None
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
