"""The pieces 07 adds on top of 01's chunkers and retrievers."""

from chunkers import (
    Article,
    Chunk,
    article_chunks,
    page_chunks,
    small_to_big,
    strip_prefix,
    with_context,
)
from retrievers import parent_rank, rrf


def _arts():
    return [
        Article(1, "第一章 總則", "", "甲\n乙", 0, 10),
        Article(2, "第一章 總則", "第一節 x", "丙", 10, 20),
    ]


def test_strip_prefix_drops_only_the_header_line():
    cs = article_chunks(_arts())
    assert cs[0].text.startswith("第一章 總則 /")
    s = strip_prefix(cs)
    assert s[0].text == "甲\n乙" and s[0].labels == ["1"] and s[1].text == "丙"


def test_with_context_prepends_by_article_and_index():
    cs = article_chunks(_arts())
    out = with_context(cs, {"1:0": "講甲乙的條", "2:1": "講丙的條"})
    assert out[0].text.startswith("講甲乙的條\n第一章") and out[1].text.startswith("講丙的條\n")
    assert with_context(cs, {})[0].text == cs[0].text


def test_page_chunks_label_an_article_only_when_the_page_holds_a_real_piece_of_it():
    body1 = "甲" * 30
    body2 = "丙" * 30
    arts = [Article(1, "", "", body1, 0, 40), Article(2, "", "", body2, 40, 80)]
    pages = [
        "第 1 條\n" + body1[:20],  # 20 of 30: more than half
        body1[20:] + "\n第 2 條\n" + body2[:3],  # 10 of 1 is not enough; header + 3 of 2 is not
        "",
        body2[3:],  # 27 of 30
    ]
    ps = page_chunks(pages, arts)
    assert [p.labels for p in ps] == [["1"], [], ["2"]]
    assert ps[2].meta["page"] == 3


def test_small_to_big_cuts_children_inside_their_parent():
    parents = [Chunk("第 1 條\n甲甲甲\n乙乙乙\n丙丙丙", ["1"]), Chunk("第 2 條\n丁", ["2"])]
    children, owner = small_to_big(parents, size=9)
    assert [c.text for c in children] == ["第 1 條\n甲甲甲", "乙乙乙\n丙丙丙", "第 2 條\n丁"]
    assert owner == [0, 0, 1] and children[1].labels == ["1"]
    assert parent_rank([2, 1, 1, 0], owner) == [1, 0]
    long_line, _ = small_to_big([Chunk("x" * 25, ["9"])], size=10)
    assert [len(c.text) for c in long_line] == [10, 10, 5]


def test_rrf_prefers_items_high_in_both_lists():
    assert rrf([[1, 2, 3], [3, 1, 2]])[:2] == [1, 3]
    assert rrf([[1, 2], [3]], k=1) == [1, 3, 2]
