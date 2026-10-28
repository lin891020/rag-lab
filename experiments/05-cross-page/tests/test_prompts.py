import json

from prompts import CONDITIONS, SCHEMA, SYSTEM, user_prompt


def test_three_evidence_shapes_plus_closed_book():
    assert CONDITIONS["closed-book"] is None
    assert list(CONDITIONS) == ["closed-book", "page", "pages", "merged"]
    assert SYSTEM and list(SCHEMA["properties"]) == ["quote", "value", "unit"]


def test_prompt_differs_only_in_evidence():
    a = user_prompt("幾安培？", evidence="| a |", label="表一")
    b = user_prompt("幾安培？", evidence="| b |", label="表一")
    assert a.split("依據")[0] == b.split("依據")[0] and a.split("問題：")[1] == b.split("問題：")[1]
    assert json.dumps(SCHEMA, ensure_ascii=False) in a
    assert "不提供" in user_prompt("幾安培？", evidence=None, label="")
