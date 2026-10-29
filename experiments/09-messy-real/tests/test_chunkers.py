"""01's article parser on gazette text (expected to find nothing) and the Chinese-numeral one."""

from chunkers import Chunk, cn_to_int, parse_articles, parse_articles_cn, with_prefix

GAZETTE = """行政院公報 第030卷 第206期 20241105 農業環保篇
農業部令 中華民國113年11月5日
修正「漁船船員管理規則」部分條文
第 六 條 學校應主動公開下列資訊：
一、學校之組織、職掌。
二、學校校務經營計畫。
第十四條之一　漁船船員手冊屆滿申請換發者，應填具申請書。
一、最近一年內二吋半身正面脫帽照片四張。
第二十一條 財產保管人或使用人，因故意或過失致財產毀損者，應負賠償責任。
第一百零八條 本規則自發布日施行。
"""


def test_01_parser_finds_no_article_in_gazette_text():
    assert parse_articles(GAZETTE) == []


def test_cn_to_int():
    assert [
        cn_to_int(s) for s in ("六", "十", "十四", "二十一", "三十", "一百零八", "一百二十")
    ] == [6, 10, 14, 21, 30, 108, 120]


def test_cn_parser_finds_articles_with_spaces_and_suffixes():
    arts = parse_articles_cn(GAZETTE)
    assert [a.label for a in arts] == ["6", "14-1", "21", "108"]
    assert arts[0].body.startswith("學校應主動公開下列資訊：")
    assert "二、學校校務經營計畫。" in arts[0].body
    assert arts[1].body.startswith("漁船船員手冊屆滿申請換發者")
    assert arts[3].body == "本規則自發布日施行。"


def test_cn_parser_ignores_article_mentions_inside_a_line():
    text = "第 一 條 本辦法依第三條規定訂定之。\n依第二十條辦理。\n第 二 條 甲。\n"
    assert [a.label for a in parse_articles_cn(text)] == ["1", "2"]


def test_with_prefix_puts_the_law_name_first_and_keeps_labels():
    chunks = [Chunk("第 1 條\n甲", ["1"], 0, 5), Chunk("第 2 條\n乙", ["2"], 5, 10)]
    out = with_prefix(chunks, "漁船船員管理規則")
    assert [c.text for c in out] == [
        "《漁船船員管理規則》\n第 1 條\n甲",
        "《漁船船員管理規則》\n第 2 條\n乙",
    ]
    assert [c.labels for c in out] == [["1"], ["2"]]
