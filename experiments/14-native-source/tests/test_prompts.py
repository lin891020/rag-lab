from prompts import CONDITIONS, SCHEMAS, user_prompt


def test_conditions_pair_each_chunking_across_sources():
    assert CONDITIONS["closed-book"] is None and CONDITIONS["oracle"] == "oracle"
    assert CONDITIONS["pdf/fixed-300"] == ("pdf", "fixed-300", False)
    assert CONDITIONS["html/fixed-300"] == ("html", "fixed-300", False)
    assert CONDITIONS["pdf/points"] == ("pdf", "points", True)
    assert CONDITIONS["html+att/points"] == ("html+att", "points", True)


def test_articles_prompt_mentions_points():
    p = user_prompt("問題？", qtype="articles", evidence="原文", label="節錄")
    assert "第四點" in p and '"4"' in p and "articles" in SCHEMAS["articles"]["properties"]
