import importlib.util
import json
from pathlib import Path

from prompts import LISTWISE_SCHEMA, SCHEMAS, listwise_prompt, user_prompt


def _p09():
    path = Path(__file__).parents[2] / "09-messy-real" / "prompts.py"
    spec = importlib.util.spec_from_file_location("prompts09", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_reading_prompt_is_09s_word_for_word():
    p09 = _p09()
    for t in ("value", "articles", "which"):
        assert user_prompt("問題？", qtype=t, evidence="原文", label="節錄") == p09.user_prompt(
            "問題？", qtype=t, evidence="原文", label="節錄"
        )
        assert SCHEMAS[t] == p09.SCHEMAS[t]


def test_listwise_prompt_numbers_the_passages_and_asks_for_an_order():
    p = listwise_prompt("問題？", ["甲" * 10, "乙" * 10])
    assert "[1]" in p and "[2]" in p and "問題？" in p
    assert LISTWISE_SCHEMA["properties"]["order"]["items"]["type"] == "integer"
    assert json.dumps(LISTWISE_SCHEMA, ensure_ascii=False) in p
