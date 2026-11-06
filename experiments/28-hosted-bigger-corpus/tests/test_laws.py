"""The five added laws: page -> text -> article units, labels that can never be a gold label,
and the 《法規名》 prefix that changes text only."""

import pytest

import laws
from chunkers import Chunk

PAGE = """<html><body>
<table><tr><th>法規名稱：</th><td><a id="hlLawName" href="x">輸配電設備裝置規則</a></td></tr>
<tr><th>修正日期：</th><td>民國 106 年 10 月 24 日 </td></tr></table>
<div class="law-reg-content">
<div>第 一 章 總則</div><div>第 一 節 通則</div>
<div>第 1 條</div><div>本規則依電業法訂定之。</div>
<div>第 二 節 用詞定義</div>
<div>本條文有附件 第 7 條</div><div>用詞定義如下：</div><div>一、供電線路。</div>
<div>第 7-1 條</div><div>導線之最小線徑如附表五十。</div>
</div></body></html>"""


def test_law_text_keeps_every_article_head_on_its_own_line():
    text = laws.law_text(PAGE)
    assert "本條文有附件" not in text
    assert [a.num for a in laws.parse_law(text)] == ["1", "7", "7-1"]


def test_part_headings_and_the_source_line_stay_out_of_article_bodies():
    text = "第 一 編 總則\n第 一 章 通則\n第 1 條\n本文。\n第 二 編 消防設計\n第 一 章 通則\n第 2 條\n本文二。\n資料來源：全國法規資料庫"
    arts = laws.parse_law(
        laws.law_text('<div class="law-reg-content">' + text.replace("\n", "<br>") + "</div>")
    )
    assert [a.body for a in arts] == ["本文。", "本文二。"]
    assert arts[1].chapter == "第二編 消防設計 第一章 通則"


def test_page_meta_reads_name_and_amendment_date():
    assert laws.page_meta(PAGE) == ("輸配電設備裝置規則", "民國 106 年 10 月 24 日")


def test_article_units_carry_chapter_section_and_a_law_scoped_label():
    units = laws.law_units("J0030019", laws.parse_law(laws.law_text(PAGE)))
    assert [u.labels for u in units] == [["J0030019:1"], ["J0030019:7"], ["J0030019:7-1"]]
    assert units[1].text.startswith("第一章 總則 / 第二節 用詞定義 / 第 7 條\n")
    assert units[2].text.endswith("導線之最小線徑如附表五十。")


def test_long_articles_split_at_lines_and_repeat_the_header():
    body = "\n".join("款" * 300 for _ in range(5))
    units = laws.law_units("X", [laws.LawArticle("3", "", "", body)], max_len=1000)
    assert len(units) == 2 and all(u.text.startswith(" /  / 第 3 條\n") for u in units)
    assert all(u.labels == ["X:3"] for u in units)


def test_added_labels_never_equal_a_base_label():
    base = {"25", "表五○", "card:表五○", "表二五～二|絞線 325"}
    added = laws.scoped_tables("J0030019", {"表五○": "表五○：最小線徑\n銅 22 14"})
    assert set(added) == {"J0030019:表五○"}
    assert not ({"J0030019:25", *added, "card:J0030019:表五○"} & base)


def test_with_name_prefixes_text_and_keeps_labels():
    c = Chunk("第 25 條\n本文", ["25"], meta={"article": 25})
    (n,) = laws.with_name([c], "用戶用電設備裝置規則")
    assert n.text == "《用戶用電設備裝置規則》\n第 25 條\n本文"
    assert n.labels == ["25"] and n.meta == {"article": 25} and c.text == "第 25 條\n本文"


@pytest.mark.skipif(not laws.LAWS_DIR.exists(), reason="run fetch_corpus.py first")
def test_each_fetched_law_parses_to_the_article_count_of_its_page():
    manifest = laws.load_manifest()
    assert set(manifest) == set(laws.LAWS)
    for pcode, m in manifest.items():
        arts = laws.parse_law((laws.LAWS_DIR / f"{pcode}.txt").read_text("utf-8"))
        assert len(arts) == m["articles_on_page"], pcode
        assert len({a.num for a in arts}) == len(arts), pcode
        for tid, text in laws.load_tables(pcode).items():
            assert text.startswith(tid), (pcode, tid)
