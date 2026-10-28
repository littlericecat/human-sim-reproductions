"""The memory stream and its retrieval function (paper §4.1).

Each memory object has a natural-language description, a creation time, a most-recent-access time, an
importance score (1–10, asked from the language model at creation) and an embedding. Retrieval scores every
memory as recency + importance + relevance after min–max scaling each to [0, 1] (all weights 1.0):
  recency   = 0.995 ** (game hours since the memory was last accessed)
  importance = the 1–10 score
  relevance  = cosine similarity between the memory's embedding and the query's embedding.
The top-ranked memories are returned and their last-access time is updated.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Any, Callable, Iterable, Optional

import numpy as np

DECAY = 0.995
KINDS = ("observation", "reflection", "plan", "dialogue")


@dataclasses.dataclass
class MemoryNode:
    id: int
    kind: str
    text: str
    created: dt.datetime
    last_accessed: dt.datetime
    importance: int
    embedding: Optional[np.ndarray] = None
    pointers: list[int] = dataclasses.field(default_factory=list)  # evidence for reflections
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "text": self.text, "created": self.created.isoformat(),
                "last_accessed": self.last_accessed.isoformat(), "importance": self.importance, "pointers": list(self.pointers), "meta": self.meta}


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


class MemoryStream:
    def __init__(self, embed: Callable[[str], np.ndarray]) -> None:
        self.nodes: list[MemoryNode] = []
        self.embed = embed
        self.importance_since_reflection = 0

    def add(self, kind: str, text: str, now: dt.datetime, importance: int, pointers: Iterable[int] = (), **meta: Any) -> MemoryNode:
        assert kind in KINDS, kind
        node = MemoryNode(id=len(self.nodes), kind=kind, text=text, created=now, last_accessed=now, importance=int(importance),
                          embedding=self.embed(text), pointers=list(pointers), meta=dict(meta))
        self.nodes.append(node)
        if kind in ("observation", "dialogue"):
            self.importance_since_reflection += node.importance
        return node

    def recent(self, n: int = 100) -> list[MemoryNode]:
        return self.nodes[-n:]

    def scores(self, query: str, now: dt.datetime) -> list[tuple[float, MemoryNode]]:
        if not self.nodes:
            return []
        q = self.embed(query)
        rec = [DECAY ** max(0.0, (now - m.last_accessed).total_seconds() / 3600.0) for m in self.nodes]
        imp = [float(m.importance) for m in self.nodes]
        rel = [float(np.dot(q, m.embedding)) if m.embedding is not None else 0.0 for m in self.nodes]
        rec, imp, rel = minmax(rec), minmax(imp), minmax(rel)
        return [(r + i + v, m) for r, i, v, m in zip(rec, imp, rel, self.nodes)]

    def retrieve(self, query: str, now: dt.datetime, k: int = 10, kinds: Optional[Iterable[str]] = None) -> list[MemoryNode]:
        ranked = sorted(self.scores(query, now), key=lambda t: (-t[0], -t[1].id))
        if kinds is not None:
            ks = set(kinds)
            ranked = [t for t in ranked if t[1].kind in ks]
        out = [m for _, m in ranked[:k]]
        for m in out:
            m.last_accessed = now
        return out

    def find(self, predicate: Callable[[MemoryNode], bool]) -> list[MemoryNode]:
        return [m for m in self.nodes if predicate(m)]

    def as_list(self) -> list[dict[str, Any]]:
        return [m.as_dict() for m in self.nodes]
