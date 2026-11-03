"""Every condition sends the same prompt; only the evidence block (the table's rendering) differs."""

import json

from prompts import CONDITIONS, SCHEMA, SYSTEM, user_prompt


def test_conditions():
    assert list(CONDITIONS) == [
        "clean",
        "glm2-s100",
        "glm2-s83",
        "glm2-s67",
        "glm2-s50",
        "glm2-s33",
    ]
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
