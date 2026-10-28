import json

from prompts import CONDITIONS, SCHEMA, SYSTEM, TOP_K, user_prompt


def test_conditions_form_a_clean_2x2_plus_closed_book():
    assert CONDITIONS["closed-book"] is None
    assert {CONDITIONS[c] for c in ("md", "md+range", "md+rows", "md+range+rows")} == {
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    }
    assert TOP_K == 3 and SYSTEM and list(SCHEMA["properties"]) == ["quote", "value", "unit"]


def test_prompt_differs_only_in_evidence():
    a = user_prompt("幾安培？", evidence="| a |", label="表一")
    b = user_prompt("幾安培？", evidence="| b |", label="表一")
    assert a.split("依據")[0] == b.split("依據")[0] and a.split("問題：")[1] == b.split("問題：")[1]
    assert json.dumps(SCHEMA, ensure_ascii=False) in a
    assert "不提供" in user_prompt("幾安培？", evidence=None, label="")
