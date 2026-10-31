"""12's CrossEncoderReranker (bge-reranker-v2-m3), the only reranker 17 uses: pointwise scores
per (question, chunk text), cached in data/17-table-retrieval/<name>.json. The `scorer` hook
exists so tests run without torch.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

from chunkers import Chunk

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data" / "17-table-retrieval"
Scorer = Callable[[list[tuple[str, str]]], list[float]]


class ScoreCache:
    def __init__(self, path: Path | None):
        self.path = path
        self.scores: dict[str, float] = {}
        self._dirty = False
        if path and path.exists():
            self.scores = json.loads(path.read_text("utf-8"))

    @staticmethod
    def key(question: str, text: str) -> str:
        return hashlib.sha256((question + "\x00" + text).encode("utf-8")).hexdigest()[:32]

    def save(self) -> None:
        if self.path and self._dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.scores), "utf-8")
            self._dirty = False


class CrossEncoderReranker:
    """Pointwise scores, one per (question, chunk); higher = more relevant."""

    def __init__(self, name: str, *, scorer: Scorer | None = None, cache_path: Path | None = None):
        self.name = name
        self._scorer = scorer
        self.cache = ScoreCache(
            cache_path if cache_path else DATA / f"{name.replace('/', '_')}.json"
        )

    def scorer(self) -> Scorer:
        if self._scorer is None:
            self._scorer = _load_cross_encoder(self.name)
        return self._scorer

    def rerank(self, question: str, chunks: list[Chunk]) -> tuple[list[int], float]:
        t0 = time.perf_counter()
        keys = [self.cache.key(question, c.text) for c in chunks]
        todo = [(i, k) for i, k in enumerate(keys) if k not in self.cache.scores]
        if todo:
            scores = self.scorer()([(question, chunks[i].text) for i, _ in todo])
            for (_, k), s in zip(todo, scores, strict=True):
                self.cache.scores[k] = float(s)
            self.cache._dirty = True
        s = [self.cache.scores[k] for k in keys]
        order = sorted(range(len(chunks)), key=lambda i: (-s[i], i))
        return order, time.perf_counter() - t0

    def save(self) -> None:
        self.cache.save()


def _load_cross_encoder(name: str) -> Scorer:
    from sentence_transformers import CrossEncoder  # the optional `rerank` group

    model = CrossEncoder(name, max_length=2048)
    logger.info("cross-encoder loaded", extra={"ctx": {"name": name, "device": str(model.device)}})
    return lambda pairs: [float(x) for x in model.predict(pairs, batch_size=8)]
