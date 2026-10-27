"""Hit metrics and bootstrap intervals. All inputs are plain lists so they are easy to test."""

from __future__ import annotations

import numpy as np

from chunkers import Chunk


def hit_at_k(ranked: list[Chunk], gold: list[str], *, k: int) -> bool:
    g = set(gold)
    return any(g & set(c.labels) for c in ranked[:k])


def hit_at_budget(ranked: list[Chunk], gold: list[str], *, budget: int) -> bool:
    """Walk the ranking, keep every chunk that still fits whole within `budget` characters."""
    g = set(gold)
    used = 0
    for c in ranked:
        if used + len(c.text) > budget:
            break
        used += len(c.text)
        if g & set(c.labels):
            return True
    return False


def bootstrap_ci(values: list[float], *, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """(mean, lo95, hi95) by resampling the questions."""
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff_ci(a: list[float], b: list[float], *, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """(mean(a-b), lo95, hi95): the same resampled questions are used for both arms."""
    d = [float(x) - float(y) for x, y in zip(a, b, strict=True)]
    return bootstrap_ci(d, n=n, seed=seed)
