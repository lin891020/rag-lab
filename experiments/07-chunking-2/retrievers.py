"""Three retrievers over a list of Chunk: dense (bge-m3 cosine), BM25 (char bigrams), random."""

from __future__ import annotations

import hashlib
import logging
import math
import random
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np

from chunkers import Chunk

logger = logging.getLogger(__name__)

EmbedFn = Callable[[list[str]], list[list[float]]]


class EmbedCache:
    """sha256(text) -> vector, persisted as one .npz. Keyed per model by using one file per digest."""

    def __init__(self, path: Path):
        self.path = path
        self.vecs: dict[str, np.ndarray] = {}
        if path.exists():
            z = np.load(path, allow_pickle=False)
            self.vecs = dict(zip(z["keys"].tolist(), z["vecs"]))
        self._dirty = False

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def fetch(self, texts: list[str], embed: EmbedFn, batch: int = 64) -> np.ndarray:
        missing = sorted({t for t in texts if self.key(t) not in self.vecs})
        for i in range(0, len(missing), batch):
            part = missing[i : i + batch]
            for t, v in zip(part, embed(part)):
                self.vecs[self.key(t)] = np.asarray(v, dtype=np.float32)
        if missing:
            self._dirty = True
        logger.info("embeddings", extra={"ctx": {"n": len(texts), "computed": len(missing)}})
        return np.stack([self.vecs[self.key(t)] for t in texts])

    def save(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self.vecs)
        np.savez(self.path, keys=np.array(keys), vecs=np.stack([self.vecs[k] for k in keys]))
        self._dirty = False


class DenseIndex:
    def __init__(self, matrix: np.ndarray, embed: EmbedFn, cache: EmbedCache):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.matrix = matrix / np.where(norms == 0, 1, norms)
        self.embed = embed
        self.cache = cache

    @classmethod
    def build(cls, chunks: list[Chunk], embed: EmbedFn, cache: EmbedCache) -> DenseIndex:
        return cls(cache.fetch([c.text for c in chunks], embed), embed, cache)

    def search(self, query: str, k: int) -> list[int]:
        q = self.cache.fetch([query], self.embed)[0]
        q = q / (np.linalg.norm(q) or 1)
        scores = self.matrix @ q
        return [int(i) for i in np.argsort(-scores, kind="stable")[:k]]


_STRIP = re.compile(r"\s+")


def bigrams(text: str) -> list[str]:
    s = _STRIP.sub("", text)
    return [s[i : i + 2] for i in range(len(s) - 1)] or [s]


class BM25Index:
    """Okapi BM25 over character bigrams; no dictionary, so it works on any Chinese text."""

    def __init__(self, chunks: list[Chunk], *, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(bigrams(c.text)) for c in chunks]
        self.lens = np.array([sum(d.values()) for d in self.docs], dtype=np.float32)
        self.avg = float(self.lens.mean()) if len(self.lens) else 0.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int) -> list[int]:
        scores = np.zeros(len(self.docs), dtype=np.float32)
        for t in set(bigrams(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, d in enumerate(self.docs):
                tf = d.get(t, 0)
                if tf:
                    denom = tf + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg)
                    scores[i] += idf * tf * (self.k1 + 1) / denom
        return [int(i) for i in np.argsort(-scores, kind="stable")[:k]]


def random_rankings(*, n_chunks: int, k: int, draws: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [rng.sample(range(n_chunks), k) for _ in range(draws)]


# -- 07 additions ------------------------------------------------------------------------------


def rrf(rankings: list[list[int]], *, k: int = 60, depth: int = 100) -> list[int]:
    """Reciprocal rank fusion of several rankings (Cormack et al. 2009), ties by first ranking."""
    score: dict[int, float] = {}
    for r in rankings:
        for rank, i in enumerate(r[:depth]):
            score[i] = score.get(i, 0.0) + 1.0 / (k + rank + 1)
    order = sorted(
        score, key=lambda i: (-score[i], rankings[0].index(i) if i in rankings[0] else depth)
    )
    return order


class HybridIndex:
    """dense + bm25 fused by RRF."""

    def __init__(self, dense: DenseIndex, bm25: BM25Index, *, depth: int = 100):
        self.dense, self.bm25, self.depth = dense, bm25, depth

    def search(self, query: str, k: int) -> list[int]:
        d = self.dense.search(query, self.depth)
        b = self.bm25.search(query, self.depth)
        return rrf([d, b], depth=self.depth)[:k]


def parent_rank(child_rank: list[int], owner: list[int]) -> list[int]:
    """Small-to-big: children ranking -> parents in first-seen order, orphans dropped."""
    seen: list[int] = []
    for i in child_rank:
        j = owner[i]
        if j >= 0 and j not in seen:
            seen.append(j)
    return seen
