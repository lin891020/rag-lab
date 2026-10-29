import importlib.util
import json
from pathlib import Path

from prompts import EVIDENCE, PROMPTS, REAL_CELL, SCHEMAS, user_prompt


def _p09():
    path = Path(__file__).parents[2] / "09-messy-real" / "prompts.py"
    spec = importlib.util.spec_from_file_location("prompts09", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_p0_is_09s_prompt_word_for_word():
    p09 = _p09()
    for t in ("value", "articles", "which"):
        for ev in ("原文", None):
            assert user_prompt("問題？", qtype=t, evidence=ev, label="節錄", prompt="P0") == (
                p09.user_prompt("問題？", qtype=t, evidence=ev, label="節錄")
            )
        assert SCHEMAS["P0"][t] == p09.SCHEMAS[t]


def test_p1_puts_found_first_and_says_what_to_do_when_missing():
    for t in ("value", "articles", "which"):
        keys = list(SCHEMAS["P1"][t]["properties"])
        assert keys[0] == "found" and "found" in SCHEMAS["P1"][t]["required"]
        p = user_prompt("問題？", qtype=t, evidence="原文", label="節錄", prompt="P1")
        assert (
            "found" in p and "false" in p and json.dumps(SCHEMAS["P1"][t], ensure_ascii=False) in p
        )


def test_grid_constants():
    assert PROMPTS == ["P0", "P1"]
    assert EVIDENCE == ["gold", "same-doc", "other-doc", "none", "real"]
    assert REAL_CELL == ("ocrrows", "article-cn+pre", True)
