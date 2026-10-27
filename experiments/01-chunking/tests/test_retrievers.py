from pathlib import Path

from chunkers import Chunk
from retrievers import BM25Index, DenseIndex, EmbedCache, random_rankings

DOCS = [
    Chunk("接地電極導線之線徑應符合表九三～一規定", ["a"]),
    Chunk("乾衣機每具以二千瓦特計算負載", ["b"]),
    Chunk("進屋線與燃氣供給管路間應保持一米以上之間隔", ["c"]),
]


def test_bm25_ranks_the_lexically_matching_chunk_first():
    idx = BM25Index(DOCS)
    assert idx.search("乾衣機的負載怎麼算", k=3)[0] == 1
    assert idx.search("進屋線跟瓦斯管要隔多遠", k=3)[0] == 2


def test_bm25_returns_at_most_k():
    assert len(BM25Index(DOCS).search("導線", k=2)) == 2


def test_dense_ranks_by_cosine_and_uses_cache(tmp_path: Path):
    table = {"甲": [1.0, 0.0], "乙": [0.0, 1.0], "丙": [0.7, 0.7], "q": [0.9, 0.1]}
    calls: list[str] = []

    def fake_embed(texts: list[str]) -> list[list[float]]:
        calls.extend(texts)
        return [table[t] for t in texts]

    chunks = [Chunk("甲", ["1"]), Chunk("乙", ["2"]), Chunk("丙", ["3"])]
    cache = EmbedCache(tmp_path / "cache.npz")
    idx = DenseIndex.build(chunks, fake_embed, cache)
    assert idx.search("q", k=3) == [0, 2, 1]
    assert sorted(calls) == ["q", "丙", "乙", "甲"]
    cache.save()
    calls.clear()
    DenseIndex.build(chunks, fake_embed, EmbedCache(tmp_path / "cache.npz"))
    assert calls == []  # everything came from the cache


def test_random_rankings_are_seeded_and_cover_k():
    a = random_rankings(n_chunks=50, k=5, draws=3, seed=1)
    b = random_rankings(n_chunks=50, k=5, draws=3, seed=1)
    assert a == b and len(a) == 3 and all(len(r) == 5 and len(set(r)) == 5 for r in a)
