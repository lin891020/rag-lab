import json

from prompts import CONDITIONS, SCHEMAS, user_prompt


def test_each_type_has_a_schema_and_the_prompt_embeds_it():
    for t in ("value", "articles", "which"):
        p = user_prompt("問題？", qtype=t, evidence="原文", label="節錄")
        assert json.dumps(SCHEMAS[t], ensure_ascii=False) in p and "原文" in p and "問題？" in p


def test_closed_book_prompt_has_no_evidence_block():
    p = user_prompt("問題？", qtype="value", evidence=None, label="")
    assert "<<<" not in p and "不提供原文" in p


def test_conditions_point_at_retrieval_cells():
    assert CONDITIONS["closed-book"] is None and CONDITIONS["oracle"] == "oracle"
    assert CONDITIONS["base"] == ("base", "article", False)
    assert CONDITIONS["all"] == ("rows", "article-cn+pre", True)
