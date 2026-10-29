from chunkers import Chunk
from run_retrieval import budget_has_gold, doc_chunks

TEXT_POINTS = (
    "修正「X要點」第二點。\n一、目的：" + "甲" * 400 + "\n二、對象：\n（一）" + "乙" * 200 + "\n"
)
TEXT_ROWS = "修正「Y收費標準」第三條附表。\n項目：犬貓收容費｜單位：每日｜收費標準：200元\n項目：安樂死費｜單位：每隻｜收費標準：500元\n"


def test_doc_chunks_per_chunking_and_kind():
    assert all(len(c.text) <= 1000 for c in doc_chunks(TEXT_POINTS, "fixed-1000"))
    small = doc_chunks(TEXT_POINTS, "fixed-300")
    assert len(small) >= 3 and all(len(c.text) <= 300 for c in small)
    pts = doc_chunks(TEXT_POINTS, "points", kind="points")
    assert [c.labels for c in pts] == [["pre"], ["1"], ["2"]]
    rows = doc_chunks(TEXT_ROWS, "rows", kind="tables")
    assert [c.labels for c in rows if c.labels == ["row"]] and len(rows) == 3
    assert rows[1].text == "項目：犬貓收容費｜單位：每日｜收費標準：200元"
    # rows on a 要點 document is points; points on a document without 點 is fixed-300
    assert [c.labels for c in doc_chunks(TEXT_POINTS, "rows", kind="points")] == [
        ["pre"],
        ["1"],
        ["2"],
    ]
    assert all(c.labels == [] for c in doc_chunks("沒有編號的文字。" * 50, "points", kind="points"))


def test_budget_has_gold_follows_the_reading_rule():
    gold = {"doc": "A", "quote": "並檢附下列文件"}
    big = Chunk("並檢附下列文件" + "乙" * 1200, [], 0, 0, {"doc": "A"})
    assert budget_has_gold([big], gold)  # first chunk cut to the budget still holds the quote
    assert not budget_has_gold([Chunk("甲" * 900, [], 0, 0, {"doc": "B"}), big], gold)
