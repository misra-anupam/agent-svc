"""Semantic answer-cache index: find a previously answered *similar* question.

Maps an embedded question to the exact cache key that holds its answer. Entries live in a
bounded per-partition list scanned with cosine similarity, in process memory, so it is
per-instance (a pgvector-backed index is the multi-instance upgrade). A partition is the hash of
(agent, version, tenant, owner, earlier conversation), so a hit can only come from the same
agent version, the same cache scope and the same context.

Risk: similar is not identical ("cancel my order" vs "cancel my orders"). Off by default; keep
the threshold high and scope narrow. See docs/security.md (cache poisoning / LLM09).
"""

from __future__ import annotations

import math
from collections import OrderedDict

from agent_svc.domain.ports import Embedder


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class SemanticIndex:
    def __init__(self, embedder: Embedder, *, threshold: float, max_entries: int = 500) -> None:
        self._embedder, self._threshold, self._max = embedder, threshold, max_entries
        self._parts: dict[str, OrderedDict[str, list[float]]] = {}

    async def find(self, partition: str, text: str) -> str | None:
        entries = self._parts.get(partition)
        if not entries:
            return None
        q = _unit((await self._embedder.embed([text]))[0])
        best_key, best = None, self._threshold
        for key, vec in entries.items():
            score = sum(a * b for a, b in zip(q, vec, strict=True))
            if score >= best:
                best_key, best = key, score
        return best_key

    async def add(self, partition: str, text: str, key: str) -> None:
        vec = _unit((await self._embedder.embed([text]))[0])
        entries = self._parts.setdefault(partition, OrderedDict())
        entries[key] = vec
        entries.move_to_end(key)
        while len(entries) > self._max:
            entries.popitem(last=False)
        if len(self._parts) > 10_000:  # bound the number of partitions too
            self._parts.pop(next(iter(self._parts)))
