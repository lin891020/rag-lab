from chunkers import Chunk
from run_read import leak, other_doc_chunk, same_doc_window


def _c(text, doc):
    return Chunk(text, [], 0, 0, {"doc": doc})


def test_same_doc_window_skips_past_the_quote_and_never_contains_it():
    text = "甲" * 3000 + "每案補助新臺幣 5,000 元整" + "乙" * 3000
    win = same_doc_window(text, "每案補助新臺幣5,000元整", budget=1000)
    assert win is not None and len(win) == 1000 and "5,000" not in win and win[0] == "乙"


def test_same_doc_window_wraps_to_the_start_when_the_quote_is_near_the_end():
    text = "甲" * 3000 + "出具後五日內送交保險人" + "乙" * 200
    win = same_doc_window(text, "出具後五日內送交保險人", budget=1000)
    assert win is not None and set(win) == {"甲"}


def test_same_doc_window_gives_up_on_a_document_too_short_to_avoid_the_quote():
    assert same_doc_window("短文 出具後五日內", "出具後五日內", budget=1000) is None


def test_leak_finds_the_gold_number_next_to_its_unit_in_either_script():
    assert leak("……每案最高補助 5,000 元……", "value", {"value": 5000, "unit": "元"})
    assert leak("……每案最高補助五千元……", "value", {"value": 5000, "unit": "元"})
    assert leak("……應於六十日內……", "value", {"value": 60, "unit": "日"})
    assert not leak("……第五條……五人……", "value", {"value": 5, "unit": "日"})  # 5 but not 5 日
    assert not leak("……每案最高補助 8,000 元……", "value", {"value": 5000, "unit": "元"})
    assert leak("……總額新臺幣五千萬元……", "value", {"value": 50_000_000, "unit": "元"})
    assert leak("……三萬六百元以上……", "value", {"value": 30600, "unit": "元"})


def test_leak_for_the_other_two_types():
    assert leak("修正第三條、第六條條文", "articles", {"articles": ["3", "6"]})
    assert not leak("修正第三條條文", "articles", {"articles": ["3", "6"]})
    assert leak("依「臺北市A辦法」第二條", "which", {"law": "臺北市A辦法"})
    assert not leak("依「臺北市B辦法」第二條", "which", {"law": "臺北市A辦法"})


def test_other_doc_chunk_is_the_best_ranked_chunk_from_another_document():
    ranked = [_c("甲" * 50, "A"), _c("乙" * 1500, "B"), _c("丙" * 50, "C")]
    text, doc = other_doc_chunk(ranked, "A", budget=1000)
    assert doc == "B" and len(text) == 1000 and set(text) == {"乙"}
    assert other_doc_chunk([_c("甲", "A")], "A", budget=1000) == (None, None)
