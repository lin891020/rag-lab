from score import judge, norm_article, norm_law


def test_norm_article_folds_every_spelling_to_one():
    assert [norm_article(s) for s in ("第十二條之一", "12-1", "12之1", "第 12 條之 1")] == [
        "12-1"
    ] * 4
    assert [norm_article(s) for s in ("第四點", "4", "第 四 條", 4)] == ["4"] * 4
    assert norm_article("二十六之三") == "26-3"


def test_articles_judge_compares_sets():
    exp = {"articles": ["2", "5"]}
    assert judge('{"quote": "x", "articles": ["第五條", "第二條"]}', "articles", exp)["value_ok"]
    assert not judge('{"quote": "x", "articles": ["2"]}', "articles", exp)["value_ok"]
    assert not judge('{"quote": "x", "articles": "2, 5"}', "articles", exp)["schema_ok"]


def test_which_judge_folds_quotes_and_accepts_alternates():
    exp = {"law": "臺北市A辦法", "alt": ["臺北市B辦法"]}
    assert judge('{"quote": "x", "law": "「臺北市A辦法」"}', "which", exp)["value_ok"]
    assert judge('{"quote": "x", "law": "臺北市B辦法"}', "which", exp)["value_ok"]
    assert judge('{"quote": "x", "law": "臺北市A辦法修正草案"}', "which", exp)["value_ok"]
    assert not judge('{"quote": "x", "law": "臺北市C辦法"}', "which", exp)["value_ok"]
    assert norm_law("《 臺北市 A 辦法 》") == "臺北市A辦法"


def test_value_judge_is_06s():
    exp = {"value": 8000, "unit": "元"}
    assert judge('{"quote": "x", "value": 8000, "unit": "元"}', "value", exp)["value_ok"]
    r = judge('{"quote": "x", "value": 8000, "unit": "萬元"}', "value", exp)
    assert not r["value_ok"] and r["value_only_ok"]
    assert not judge("not json", "value", exp)["parse_ok"]
