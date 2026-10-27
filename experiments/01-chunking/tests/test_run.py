from chunkers import Chunk
from retrievers import BM25Index
from run import evaluate, evaluate_random

CHUNKS = [
    Chunk("接地電極導線之線徑應符合表九三～一規定", ["93"]),
    Chunk("乾衣機每具以二千瓦特計算負載", ["55"]),
    Chunk("進屋線與燃氣供給管路間應保持一米以上之間隔", ["66"]),
]
QS = [
    {"id": "q1", "type": "article", "question": "乾衣機的負載怎麼算", "gold": ["55"]},
    {"id": "q2", "type": "article", "question": "進屋線跟瓦斯管要隔多遠", "gold": ["66"]},
    {"id": "q3", "type": "article", "question": "完全無關的問題", "gold": ["999"]},
]


def test_evaluate_one_record_per_question_with_all_metrics():
    idx = BM25Index(CHUNKS)
    rows = evaluate(CHUNKS, idx.search, QS, k=2, budgets=[10, 100])
    assert [r["qid"] for r in rows] == ["q1", "q2", "q3"]
    assert rows[0]["hit@2"] == 1.0 and rows[0]["hit@100"] == 1.0 and rows[0]["hit@10"] == 0.0
    assert rows[2]["hit@2"] == 0.0
    assert rows[0]["top"][0] == ["55"]


def test_evaluate_random_returns_fractions_between_0_and_1():
    rows = evaluate_random(CHUNKS, QS, k=1, budgets=[100], draws=50, seed=0)
    assert len(rows) == 3 and all(0.0 <= r["hit@1"] <= 1.0 for r in rows)
    assert rows[2]["hit@1"] == 0.0  # gold never present
    assert 0.1 < rows[0]["hit@1"] < 0.6  # one of three chunks is gold
