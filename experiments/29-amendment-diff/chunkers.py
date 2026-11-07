"""The unit of indexing: a text with the labels (條號) it covers. 27's dataclass, nothing else of
27's chunkers is needed here (laws.py cuts the law into articles)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    labels: list[str]
    start: int = 0
    end: int = 0
    meta: dict = field(default_factory=dict)
