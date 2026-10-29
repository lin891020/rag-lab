from attribute import PREDICTED_BY_08, classify, quote_position

H_CLEAN = {"pages": 4, "text_pages": 4, "garbage_pages": 0, "rotated_pages": 0}


def _c(**kw):
    base = dict(
        hit=False,
        doc_chunks=5,
        gold_chunks=1,
        quote_in_text=True,
        quote_before_first_article=False,
        chunking="article-cn",
        health=H_CLEAN,
        top5_docs=["A", "B"],
        doc="A",
    )
    base.update(kw)
    return classify(**base)


def test_hits_have_no_cause():
    assert _c(hit=True) is None


def test_causes_in_order():
    assert _c(doc_chunks=0) == "no_chunks"
    assert (
        _c(gold_chunks=0, quote_in_text=False, health={**H_CLEAN, "garbage_pages": 3}) == "garbage"
    )
    assert _c(gold_chunks=0, quote_in_text=False, health={**H_CLEAN, "text_pages": 1}) == "no_text"
    assert (
        _c(gold_chunks=0, quote_in_text=False, health={**H_CLEAN, "rotated_pages": 2}) == "rotated"
    )
    assert _c(gold_chunks=0, quote_in_text=False) == "extract_other"
    assert _c(gold_chunks=0, quote_before_first_article=True) == "preamble_dropped"
    assert _c(gold_chunks=0, quote_before_first_article=True, chunking="fixed-1000") == "chunk_cut"
    assert _c(gold_chunks=0) == "chunk_cut"
    assert _c(hit5=True) == "over_budget"
    assert _c(top5_docs=["B", "C"]) == "wrong_doc"
    assert _c() == "ranked_low"
    assert {"no_chunks", "no_text", "chunk_cut", "over_budget", "wrong_doc"} == PREDICTED_BY_08


def test_quote_position_sees_the_front_matter():
    text = "農業部令\n修正「漁船船員管理規則」部分條文\n第一條　本規則依法訂定之。\n"
    assert quote_position(text, "部分條文", "article-cn") == (True, True)
    assert quote_position(text, "本規則依法訂定之", "article-cn") == (True, False)
    assert quote_position(text, "不在裡面", "article-cn") == (False, False)
    assert quote_position(text, "本規則依法訂定之", "article") == (
        True,
        False,
    )  # 01's parser finds no article: nothing was dropped, so it is not "before the first article"
