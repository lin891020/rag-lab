"""27's dense retriever (bge-m3 cosine) over a list of Chunk, with its embedding cache; BM25 and
random are not copied (29 does not use them)."""

from __future__ import annotations

import hashlib
import logging
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
