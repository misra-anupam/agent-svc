from __future__ import annotations

import json
import math
from typing import Any

import asyncpg

from agent_svc.domain.ports import Embedder
from agent_svc.domain.ports.memory import MemoryHit


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class InMemoryLongTermMemory:
    def __init__(self, embedder: Embedder) -> None:
        self._e = embedder
        self._d: dict[tuple[str, str], list[tuple[list[float], str, dict[str, Any]]]] = {}

    async def remember(self, tenant: str, user_id: str, text: str, **meta: Any) -> None:
        [vec] = await self._e.embed([text])
        self._d.setdefault((tenant, user_id), []).append((vec, text, meta))

    async def recall(self, tenant: str, user_id: str, query: str, k: int = 5) -> list[MemoryHit]:
        [q] = await self._e.embed([query])
        scored = [
            MemoryHit(text, _cos(q, vec), meta)
            for vec, text, meta in self._d.get((tenant, user_id), [])
        ]
        return sorted(scored, key=lambda h: h.score, reverse=True)[:k]

    async def forget(self, tenant: str, user_id: str) -> None:
        self._d.pop((tenant, user_id), None)


def _vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


class PgVectorLongTermMemory:
    """Rows are namespaced by (tenant, user_id); every query filters on both (LLM08)."""

    def __init__(self, pool: asyncpg.Pool, embedder: Embedder) -> None:
        self._pool, self._e = pool, embedder

    async def remember(self, tenant: str, user_id: str, text: str, **meta: Any) -> None:
        [vec] = await self._e.embed([text])
        await self._pool.execute(
            "INSERT INTO long_term_memory (tenant, user_id, text, meta, embedding) "
            "VALUES ($1, $2, $3, $4::jsonb, $5::vector)",
            tenant,
            user_id,
            text,
            json.dumps(meta),
            _vec(vec),
        )

    async def recall(self, tenant: str, user_id: str, query: str, k: int = 5) -> list[MemoryHit]:
        [q] = await self._e.embed([query])
        rows = await self._pool.fetch(
            "SELECT text, meta, 1 - (embedding <=> $3::vector) AS score FROM long_term_memory "
            "WHERE tenant=$1 AND user_id=$2 ORDER BY embedding <=> $3::vector LIMIT $4",
            tenant,
            user_id,
            _vec(q),
            k,
        )
        return [MemoryHit(r["text"], float(r["score"]), json.loads(r["meta"])) for r in rows]

    async def forget(self, tenant: str, user_id: str) -> None:
        await self._pool.execute(
            "DELETE FROM long_term_memory WHERE tenant=$1 AND user_id=$2", tenant, user_id
        )
