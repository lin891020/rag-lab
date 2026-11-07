"""The comparison table -> rows. Pure parts on synthetic pages: a header word is 「第N條」 alone
(a wrapped 「第二十五條規定」 is body text), a page's first row without a header continues the
previous page's last row, a row keeps the article number each text column starts with. The real
PDF (data/, not in git): every new article 1–1,014 starts exactly one row, every old article
(1,079, 13-1 included) starts at least one, no word crosses a column border on 30 sampled pages."""

import json
import random
import re

import pytest

import compare_table as ct
import laws

PDF = ct.COMPARE_PDF


def test_head_word_is_an_article_number_alone():
    assert ct.head_num("第六條") == "6"
    assert ct.head_num("第十三條之一") == "13-1"
    assert ct.head_num("第一千零一十四條") == "1014"
    assert ct.head_num("第二十五條規定") is None  # a wrapped body line
    assert ct.head_num("第一百七十一條第一款、") == "171" and ct.head_part("第一百七十一條第一款、")
    assert not ct.head_part("第六條")
    assert ct.head_num("第二節") is None
    assert ct.mark_word("第二節") and ct.mark_word("第一章") and not ct.mark_word("第六條")


def test_fold_text_ignores_spacing_and_punctuation_spellings():
    assert ct.fold_text("二‧五公尺以上。") == ct.fold_text("二．五公尺以上") == "二五公尺以上"
    assert ct.fold_text("第 25 條（刪除）") == "第25條刪除"
    assert (
        ct.old_body({"old_text": "第十六條 低壓絕緣導線\n之安培容量。"}) == "低壓絕緣導線之安培容量"
    )


def test_cell_heads_take_the_first_word_and_a_second_article_after_a_sentence_end():
    lines = [
        (235.0, "第九十七條", "第九十七條 可撓軟線及"),
        (235.0, "可撓電纜不得使用於下", "可撓電纜不得使用於下"),
        (247.0, "一、永久性分路配線依", "一、永久性分路配線依"),
        (259.0, "第一百五十二條", "第一百五十二條"),  # a reference wrapped onto its own line
        (235.0, "規定。", "規定。"),
        (235.0, "第九十六條", "第九十六條 可撓軟線及"),  # the merged second article
        (235.0, "可撓電纜適用於下列情", "可撓電纜適用於下列情"),
    ]
    assert ct.cell_heads(lines, 230.0) == [("97", False, 0), ("96", False, 5)]
    assert ct.cell_heads(
        [(235.0, "第一百七十一條第一款、", "第一百七十一條第一款、第二款")], 230.0
    ) == [("171", True, 0)]
    # the header wrapped onto the next line, and a 款 on the next line
    wrapped = [
        (235.0, "第二百五十二條之三第二", "第二百五十二條之三第二"),
        (235.0, "項", "項 六○○伏以下單芯"),
    ]
    assert ct.cell_heads(wrapped, 230.0) == [("252-3", True, 0)]
    two_lines = [
        (235.0, "第三百九十六條之三十七", "第三百九十六條之三十七"),
        (235.0, "第二款、第三款及第六款", "第二款、第三款及第六款"),
    ]
    assert ct.cell_heads(two_lines, 230.0) == [("396-37", True, 0)]
    whole = [
        (235.0, "第三百九十六條之七十四", "第三百九十六條之七十四"),
        (235.0, "儲能系統之電池模組裝", "儲能系統之電池模組裝"),
    ]
    assert ct.cell_heads(whole, 230.0) == [("396-74", False, 0)]


def _row(
    page, new="", old="", note="", heads_new=(), heads_old=(), marks=(), first=None, split=None
):
    return {
        "page": page,
        "cells": {"new": new, "old": old, "note": note},
        "heads": {"new": list(heads_new), "old": list(heads_old)},
        "marks": list(marks),
        "first": first
        if first is not None
        else {"new": bool(heads_new or marks), "old": bool(heads_old or marks)},
        "split": split or {},
    }


def test_assemble_merges_a_continuation_and_keeps_both_sides_numbers():
    pages = [
        [
            _row(1, "修正條文", "現行條文", "說明"),
            _row(1, "第一章 總則", "第一章 總則", "章名未修正", marks=["第一章"]),
            _row(1, "第一條 甲", "第一條 甲", "未修正", ["1"], ["1"]),
            _row(1, "第二條 乙乙", "第三條 乙", "條次變更", ["2"], ["3"]),
        ],
        [
            # no header on the first line: the row above goes on; a second old article starts
            # lower in the cell and belongs to that row
            _row(
                2,
                "乙乙乙",
                "乙乙\n第五條 己",
                "酌修",
                heads_old=["5"],
                first={"new": False, "old": False},
            ),
            _row(2, "第三條 丙", "", "新增", ["3"], []),
            _row(2, "", "第二條 丁", "刪除", [], ["2"]),
            _row(2, "第四條 戊", "第四條 戊", "合併", ["4"], ["4"]),
            # a rule inside a merged article: the new text goes on, the old column starts its
            # second source and the note runs on -> the band joins the row above
            _row(2, "戊戊", "第六條 庚", "改。", [], ["6"], first={"new": False, "old": True}),
            # the old column's text runs on before its own article starts, beside a new article:
            # the leading text goes to the row above, the band is a new row
            _row(
                2,
                "第五條 壬",
                "庚庚。\n第七條 癸",
                "一、整併。",
                ["5"],
                ["7"],
                first={"new": True, "old": False},
                split={"old": ("庚庚。", "第七條 癸")},
            ),
            # a deleted article beside an empty new cell, with a fresh note: its own row
            _row(
                2, "", "第八條 子", "一、本條刪除。", [], ["8"], first={"new": False, "old": True}
            ),
            # the old column starts an article, the new text runs on, but the note starts afresh:
            # its own row (not a merge)
            _row(
                2,
                "壬壬",
                "第九條 丑",
                "一、條次變更。",
                [],
                ["9"],
                first={"new": False, "old": True},
            ),
        ],
    ]
    rows = ct.assemble(pages)
    assert [r["kind"] for r in rows] == ["structure"] + ["article"] * 8
    r2 = rows[2]
    assert r2["new_nums"] == ["2"] and r2["old_nums"] == ["3", "5"] and r2["pages"] == [1, 2]
    assert (
        r2["new_text"] == "第二條 乙乙\n乙乙乙" and r2["old_text"] == "第三條 乙\n乙乙\n第五條 己"
    )
    assert r2["note"] == "條次變更\n酌修"
    assert rows[3]["old_nums"] == [] and rows[4]["new_nums"] == []
    assert rows[5]["new_nums"] == ["4"] and rows[5]["old_nums"] == ["4", "6"]
    assert (
        rows[5]["new_text"] == "第四條 戊\n戊戊"
        and rows[5]["old_text"] == "第四條 戊\n第六條 庚\n庚庚。"
    )
    assert rows[5]["note"] == "合併\n改。"
    assert (
        rows[6]["new_nums"] == ["5"]
        and rows[6]["old_nums"] == ["7"]
        and rows[6]["old_text"] == "第七條 癸"
    )
    assert rows[7]["new_nums"] == [] and rows[7]["old_nums"] == ["8"]
    assert (
        rows[8]["new_nums"] == [] and rows[8]["old_nums"] == ["9"] and rows[8]["new_text"] == "壬壬"
    )
    assert rows[0]["marks"] == ["第一章"]  # the column-header row is dropped


@pytest.fixture(scope="module")
def rows():
    if not PDF.exists():
        pytest.skip("comparison PDF not fetched (data/)")
    if ct.ROWS.exists():
        return json.loads(ct.ROWS.read_text("utf-8"))
    return ct.assemble(ct.read_pages(PDF))


@pytest.mark.skipif(not PDF.exists(), reason="comparison PDF not fetched")
def test_every_new_article_starts_exactly_one_row(rows):
    new = [n for r in rows for n in r["new_nums"]]
    assert sorted(new, key=ct.sort_key) == [str(i) for i in range(1, 1015)]


@pytest.mark.skipif(not PDF.exists(), reason="comparison PDF not fetched")
def test_every_old_article_starts_a_row_and_no_row_has_two(rows):
    old = [n for r in rows for n in r["old_nums"]]
    want = {a.num for a in laws.parse_law(ct.OLD_TXT.read_text("utf-8"))}
    assert len(want) == 1079 and "13-1" in old
    assert set(old) == want  # one old article may start several rows (split), none is missing


def _only_headings(text: str) -> bool:
    """Empty, or nothing but 章／節 headings (a deleted article's band may carry the heading that
    follows it, 「第二節之一 出線盒…」 or 「第八章之一 (刪除)」)."""
    cn = "零〇一二三四五六七八九十百千"
    lines = [ln for ln in text.replace(" ", "").splitlines() if ln.strip()]
    i = 0
    while i < len(lines):
        if re.match(rf"^第[{cn}]+[章節款](之[{cn}]+)?", lines[i]):
            i += 1  # the heading and its (possibly wrapped) title
            while i < len(lines) and not re.match(rf"^第[{cn}]+[章節款條]", lines[i]):
                i += 1
            continue
        return False
    return True


@pytest.mark.skipif(not PDF.exists(), reason="comparison PDF not fetched")
def test_an_added_row_has_an_empty_old_cell_and_a_deleted_row_an_empty_new_cell(rows):
    for r in rows:
        if r["kind"] != "article":
            continue
        if not r["old_nums"]:
            assert _only_headings(r["old_text"]), (r["new_nums"], r["pages"], r["old_text"][:40])
        if not r["new_nums"]:
            assert _only_headings(r["new_text"]), (r["old_nums"], r["pages"], r["new_text"][:40])


@pytest.mark.skipif(not PDF.exists(), reason="comparison PDF not fetched")
def test_old_cells_match_the_old_version_text_at_least_90_percent(rows):
    old = {a.num: a.body for a in laws.parse_law(ct.OLD_TXT.read_text("utf-8"))}
    a = ct.align([dict(r) for r in rows], old)
    assert a["compared"] > 900 and a["rate"] >= 0.90, a


@pytest.mark.skipif(not PDF.exists(), reason="comparison PDF not fetched")
def test_no_word_crosses_a_column_border_on_sampled_pages():
    import pdfplumber

    with pdfplumber.open(str(PDF)) as pdf:
        portrait = [i for i, p in enumerate(pdf.pages) if p.width < p.height]
        for i in random.Random(0).sample(portrait, 30):
            assert ct.crossing_words(pdf.pages[i]) == [], f"page {i}"
