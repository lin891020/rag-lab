"""Two spellings of one article must read as one: the old version writes 六○○伏 and 一五○, the
new one 六百伏特 and 一百五十; references (第N條, 表N～N) and list markers (一、) are masked;
punctuation and spacing are dropped. Numbers come out as values, so T3 finds rows whose values
differ, not rows whose numeral style differs."""

import pytest

import textnorm as tn


@pytest.mark.parametrize(
    "s, want",
    [
        ("六○○", 600),
        ("一五○", 150),
        ("六百", 600),
        ("一百五十", 150),
        ("二‧五", 2.5),
        ("二．五", 2.5),
        ("一千五百", 1500),
        ("〇", 0),
        ("25", 25),
        ("1,500", 1500),
        ("0.8", 0.8),
        ("八○", 80),
        ("十五", 15),
        ("三十五千", 35000),
        ("二萬五千", 25000),
        ("一千零一十四", 1014),
        ("十", 10),
        ("一萬", 10000),
    ],
)
def test_cn_value(s, want):
    assert tn.cn_value(s) == want


def test_numbers_of_masks_references_and_list_markers():
    s = "一、依第二十五條及表二五～二規定，六○○伏以下導線，間隔一‧八公尺，每一出線口一五○伏安。"
    assert tn.numbers_of(s) == [600, 1.8, 150]
    s2 = "一、依第二百五十二條之三及表二五～二規定，六百伏特以下導線，間隔一‧八公尺，每一出線口一百五十伏安。"
    assert tn.numbers_of(s2) == [600, 1.8, 150]
    assert (
        tn.numbers_of("需量因數得依表二九之三一計算，每一出線口") == []
    )  # old-style table reference


def test_canon_makes_the_two_numeral_styles_equal_and_masks_references():
    old = "第十六條 低壓導線六○○伏以下者，依表一六～三規定；間隔一‧八公尺。"
    new = "第二十五條 低壓導線六百伏特以下者，依表二五～二規定；間隔一‧八公尺。"
    a, b = tn.canon(old), tn.canon(new)
    assert a != b  # 伏 vs 伏特 is a real wording change
    old2 = "第十六條 低壓導線六○○伏特以下者，依表一六～三規定；間隔一‧八公尺。"
    assert tn.canon(old2) == b  # same values, same words, other numerals and references
    assert "第" not in tn.canon("依第十六條規定") or "＃" in tn.canon("依第十六條規定")


def test_canon_strips_the_article_header():
    assert tn.canon("第十六條 甲乙。") == tn.canon("第二十五條 甲乙。") == tn.canon("甲乙")
