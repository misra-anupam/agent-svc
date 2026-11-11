from __future__ import annotations

import json
from typing import Any

import asyncpg


class InMemoryUserMemory:
    def __init__(self) -> None:
        self._d: dict[tuple[str, str], dict[str, Any]] = {}

    async def get(self, tenant: str, user_id: str) -> dict[str, Any]:
        return dict(self._d.get((tenant, user_id), {}))

    async def merge(self, tenant: str, user_id: str, facts: dict[str, Any]) -> None:
        self._d.setdefault((tenant, user_id), {}).update(facts)

    async def delete(self, tenant: str, user_id: str) -> None:
        self._d.pop((tenant, user_id), None)


class PostgresUserMemory:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get(self, tenant: str, user_id: str) -> dict[str, Any]:
        row = await self._pool.fetchrow(
            "SELECT facts FROM user_memory WHERE tenant=$1 AND user_id=$2", tenant, user_id
        )
        return json.loads(row["facts"]) if row else {}

    async def merge(self, tenant: str, user_id: str, facts: dict[str, Any]) -> None:
        await self._pool.execute(
            """INSERT INTO user_memory (tenant, user_id, facts) VALUES ($1, $2, $3::jsonb)
               ON CONFLICT (tenant, user_id)
               DO UPDATE SET facts = user_memory.facts || EXCLUDED.facts, updated_at = now()""",
            tenant,
            user_id,
            json.dumps(facts),
        )

    async def delete(self, tenant: str, user_id: str) -> None:
        await self._pool.execute(
            "DELETE FROM user_memory WHERE tenant=$1 AND user_id=$2", tenant, user_id
        )
