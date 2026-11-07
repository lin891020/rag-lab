"""Article numbers in any spelling fold to one key: 15's norm_article with two patches, 千 (this
law has 1,014 articles) and 「第 13-1 條」 (how the database page writes a branch)."""

import pytest

from articles import cn_to_int, norm_article


@pytest.mark.parametrize(
    "s, want",
    [
        ("第一千零一十四條", "1014"),
        ("第一千條", "1000"),
        ("第一千零一條", "1001"),
        ("第 13-1 條", "13-1"),
        ("第十三條之一", "13-1"),
        ("第 25 條", "25"),
        ("第二十五條", "25"),
        ("16", "16"),
        ("13-1", "13-1"),
        ("第二百五十二條之四", "252-4"),
        ("第三百九十六條之七十七", "396-77"),
    ],
)
def test_norm_article(s, want):
    assert norm_article(s) == want


def test_cn_to_int_handles_千():
    assert cn_to_int("一千零一十四") == 1014
    assert cn_to_int("一千") == 1000
    assert cn_to_int("九百九十九") == 999
    assert cn_to_int("十") == 10


def test_unknown_strings_fold_to_themselves():
    assert norm_article("新增") == "新增"
    assert norm_article(" 刪除 ") == "刪除"
