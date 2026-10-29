from prompts import CONDITIONS, SCHEMAS, user_prompt


def test_conditions_point_at_the_three_chunkings():
    assert CONDITIONS["closed-book"] is None and CONDITIONS["oracle"] == "oracle"
    assert CONDITIONS["fixed-1000"] == ("ocr", "fixed-1000", False)
    assert CONDITIONS["fixed-300"] == ("ocr", "fixed-300", False)
    assert CONDITIONS["struct"] == ("rows", "rows", True)


def test_articles_prompt_mentions_points():
    p = user_prompt("問題？", qtype="articles", evidence="原文", label="節錄")
    assert "第四點" in p and '"4"' in p and "articles" in SCHEMAS["articles"]["properties"]
