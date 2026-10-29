"""Rerankers over a first-stage ranking, one interface: rerank(question, chunks) -> (order, seconds)
where `order` is a permutation of range(len(chunks)) (indices into the given list).

  rrf                    07's reciprocal-rank fusion (dense + bm25): a merge, not a rerank; the
                         contrast
  CrossEncoderReranker   sentence-transformers CrossEncoder (bge-reranker-v2-m3); scores cached
                         per (question, chunk text) in data/12-rerank/<name>.json
  Qwen3Reranker          Qwen3-Reranker-0.6B: a causal LM scored by P(yes) on the official
                         template (pointwise); same cache
  ListwiseReranker       an ollama chat model given the top `window` passages, asked for an
                         order (RankGPT style, JSON array); the tail keeps the dense order

The `scorer` / `chat` hooks exist so tests run without models or ollama.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

from chunkers import Chunk
from prompts import LISTWISE_SCHEMA, LISTWISE_SYSTEM, listwise_prompt

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data" / "12-rerank"
Scorer = Callable[[list[tuple[str, str]]], list[float]]


def rrf(rankings: list[list[int]], *, k: int = 60, depth: int = 100) -> list[int]:
    """07's: sum 1/(k+rank) over rankings, ties by first appearance."""
    score: dict[int, float] = {}
    first: dict[int, int] = {}
    for r in rankings:
        for rank, i in enumerate(r[:depth]):
            score[i] = score.get(i, 0.0) + 1.0 / (k + rank + 1)
            first.setdefault(i, len(first))
    return sorted(score, key=lambda i: (-score[i], first[i]))


def parse_listwise(content: str, n: int) -> tuple[list[int], bool]:
    """JSON {"order": [1-based ids]} -> 0-based permutation of range(n), and whether the output
    was clean (a permutation, nothing missing, nothing extra). Duplicates and out-of-range ids
    are dropped; missing ids follow in the original order."""
    try:
        obj = json.loads(content)
        raw = obj.get("order") if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        raw = None
    if not isinstance(raw, list):
        return list(range(n)), False
    seen: list[int] = []
    for x in raw:
        if isinstance(x, int) and not isinstance(x, bool) and 1 <= x <= n and x - 1 not in seen:
            seen.append(x - 1)
    clean = len(seen) == n and len(raw) == n
    return seen + [i for i in range(n) if i not in seen], clean


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


class Qwen3Reranker(CrossEncoderReranker):
    """Qwen/Qwen3-Reranker-0.6B scored as P(yes) over the official prompt template."""

    NAME = "Qwen/Qwen3-Reranker-0.6B"
    INSTRUCTION = "Given a question about a Taiwanese government gazette notice, judge whether the passage answers it."

    def __init__(self, *, scorer: Scorer | None = None, cache_path: Path | None = None):
        super().__init__(self.NAME, scorer=scorer, cache_path=cache_path)

    def scorer(self) -> Scorer:
        if self._scorer is None:
            self._scorer = _load_qwen3_reranker(self.NAME, self.INSTRUCTION)
        return self._scorer


def _load_qwen3_reranker(name: str, instruction: str) -> Scorer:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(device).eval()
    prefix = (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
        'and the Instruct provided. Note that the answer can only be "yes" or "no".'
        "<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    yes_id, no_id = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    logger.info("qwen3 reranker loaded", extra={"ctx": {"name": name, "device": device}})

    def score(pairs: list[tuple[str, str]]) -> list[float]:
        out: list[float] = []
        for i in range(0, len(pairs), 4):
            batch = [
                f"{prefix}<Instruct>: {instruction}\n<Query>: {q}\n<Document>: {d}{suffix}"
                for q, d in pairs[i : i + 4]
            ]
            enc = tok(batch, padding=True, truncation=True, max_length=4096, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits[:, -1, :]
            pair = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1).float()
            out += torch.log_softmax(pair, dim=1)[:, 1].exp().tolist()
        return out

    return score


class ListwiseReranker:
    """RankGPT-style: the top `window` passages (each cut to `cut` chars) numbered [1]..[n],
    the model returns an order; passages past the window keep their first-stage order."""

    def __init__(self, model: str, *, window: int = 20, cut: int = 400, chat=None, think=False):
        self.model = model
        self.window = window
        self.cut = cut
        self.think = think
        self._chat = chat
        self.bad_outputs = 0
        self.name = f"listwise:{model}"

    def chat(self, prompt: str) -> dict:
        if self._chat is not None:
            return self._chat(prompt)
        from rag_lab import ollama

        return ollama.chat(
            prompt,
            model=self.model,
            system=LISTWISE_SYSTEM,
            format=LISTWISE_SCHEMA,
            think=self.think,
            options={"temperature": 0, "seed": 0, "num_ctx": 16384, "num_predict": 256},
        )

    def rerank(self, question: str, chunks: list[Chunk]) -> tuple[list[int], float]:
        t0 = time.perf_counter()
        head = chunks[: self.window]
        if not head:
            return [], 0.0
        r = self.chat(listwise_prompt(question, [c.text[: self.cut] for c in head]))
        order, clean = parse_listwise(r.get("content", ""), len(head))
        if not clean:
            self.bad_outputs += 1
        return order + list(range(len(head), len(chunks))), time.perf_counter() - t0

    def save(self) -> None:
        return None
