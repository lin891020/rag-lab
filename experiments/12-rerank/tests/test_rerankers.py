from chunkers import Chunk
from rerankers import CrossEncoderReranker, ListwiseReranker, parse_listwise, rrf


def _c(text, doc="A"):
    return Chunk(text, [], 0, 0, {"doc": doc})


def test_parse_listwise_dedupes_drops_junk_and_fills_the_missing_in_original_order():
    assert parse_listwise('{"order": [3, 1, 3, 9, 2]}', 4) == ([2, 0, 1, 3], False)  # 1-based in
    assert parse_listwise('{"order": [2, 1, 4, 3]}', 4) == ([1, 0, 3, 2], True)
    assert parse_listwise("not json", 3) == ([0, 1, 2], False)
    assert parse_listwise('{"order": "2 > 1"}', 2) == ([0, 1], False)


def test_rrf_is_07s():
    assert rrf([[0, 1, 2], [2, 0, 1]], k=60)[:2] == [0, 2]


def test_cross_encoder_orders_by_score_with_an_injected_scorer():
    def scorer(pairs):
        return [len(doc) for _, doc in pairs]  # longer = more relevant, for the test

    rr = CrossEncoderReranker("fake", scorer=scorer)
    chunks = [_c("短"), _c("這是最長的一塊文字"), _c("中等長度")]
    order, seconds = rr.rerank("問題", chunks)
    assert order == [1, 2, 0] and seconds >= 0


def test_cross_encoder_scores_are_cached_per_question_and_chunk(tmp_path):
    calls = []

    def scorer(pairs):
        calls.append(len(pairs))
        return [1.0] * len(pairs)

    rr = CrossEncoderReranker("fake", scorer=scorer, cache_path=tmp_path / "fake.json")
    chunks = [_c("甲"), _c("乙")]
    rr.rerank("問題", chunks)
    rr.rerank("問題", chunks)
    assert calls == [2]  # second call served from the cache
    rr.save()
    rr2 = CrossEncoderReranker("fake", scorer=scorer, cache_path=tmp_path / "fake.json")
    rr2.rerank("問題", chunks)
    assert calls == [2]


def test_listwise_reranks_only_its_window_and_keeps_the_tail_in_dense_order():
    def chat(prompt, **kw):
        return {"content": '{"order": [3, 2, 1]}', "duration": 0.1}

    rr = ListwiseReranker("fake", window=3, chat=chat)
    chunks = [_c(f"塊{i}") for i in range(5)]
    order, seconds = rr.rerank("問題", chunks)
    assert order == [2, 1, 0, 3, 4]
    assert rr.bad_outputs == 0
