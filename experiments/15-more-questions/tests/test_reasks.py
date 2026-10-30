from reasks import gazette_twins, law_reasks, quote_overlap


def _q(id_, doc, type_, quote, expect):
    return {"id": id_, "doc": doc, "type": type_, "quote": quote, "expect": expect}


def test_quote_overlap_is_relative_to_the_shorter_quote():
    assert quote_overlap("委員五至二十三人", "委員五至二十三人，由副首長為召集人") == 1.0
    assert quote_overlap("完全不同", "委員五至二十三人") == 0.0


def test_a_rephrased_question_with_the_same_answer_and_quote_is_a_twin():
    old = [_q("q052", "D1", "value", "委員五至二十三人", {"value": 23, "unit": "人"})]
    new = [
        _q("q161", "D1", "value", "委員五至二十三人，由副首長", {"value": 23, "unit": "人"}),
        _q("q162", "D1", "value", "委員五至二十三人", {"value": 5, "unit": "人"}),  # other answer
        _q("q163", "D2", "value", "委員五至二十三人", {"value": 23, "unit": "人"}),  # other doc
    ]
    assert gazette_twins(old, new) == {"q161": "q052"}


def test_which_questions_of_one_document_need_only_quote_overlap():
    old = [_q("q057", "D1", "which", "應建立現場安全管控機制", {"law": "甲法"})]
    new = [
        _q("q166", "D1", "which", "應建立現場安全管控機制及應變方案", {"law": "甲法"}),
        _q("q167", "D1", "which", "高風險職務之認定", {"law": "甲法"}),
    ]
    assert gazette_twins(old, new) == {"q166": "q057"}


def test_law_reask_is_any_shared_gold():
    old = [{"id": "a01", "gold": ["16"]}, {"id": "t01", "gold": ["表二五～二"]}]
    new = [
        {"id": "a41", "gold": ["16", "17"]},
        {"id": "a42", "gold": ["18"]},
        {"id": "t18", "gold": ["表二五～二"]},
    ]
    assert law_reasks(old, new) == {"a41", "t18"}
