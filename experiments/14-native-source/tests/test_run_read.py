from chunkers import Chunk
from run_read import budget_evidence, oracle_window


def _c(text, doc):
    return Chunk(text, [], 0, 0, {"doc": doc})


def test_budget_evidence_keeps_whole_chunks_and_flags_gold():
    ranked = [_c("甲" * 600, "B"), _c("並檢附下列文件" + "乙" * 300, "A"), _c("丙" * 100, "A")]
    text, has_gold = budget_evidence(ranked, {"doc": "A", "quote": "並檢附下列文件"}, budget=1000)
    assert has_gold and "【段 2】" in text and "丙" not in text  # 600 + 307 + 100 > 1000
    text, has_gold = budget_evidence(ranked, {"doc": "A", "quote": "並檢附下列文件"}, budget=500)
    assert not has_gold and text.count("【段") == 1 and len(text) < 520  # first chunk cut


def test_oracle_window_centres_on_the_quote_ignoring_whitespace():
    text = "前面" * 400 + "出具後\n五日內送交保險\n人。" + "後面" * 400
    win = oracle_window(text, "出具後五日內送交保險人", budget=200)
    assert win is not None and "五日內送交保險" in win and len(win) == 200
    assert oracle_window(text, "沒有這句", budget=200) is None
