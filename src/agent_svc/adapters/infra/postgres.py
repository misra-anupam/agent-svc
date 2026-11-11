from __future__ import annotations

import asyncpg

from agent_svc.config.settings import PostgresSettings


class PostgresResource:
    """Shared asyncpg pool (lazy connections, min_size defaults to 0)."""

    name = "postgres"

    def __init__(self, settings: PostgresSettings) -> None:
        self._settings = settings
        self.required = settings.required
        self._pool: asyncpg.Pool | None = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PostgresResource not started")
        return self._pool

    async def start(self) -> None:
        s = self._settings
        self._pool = await asyncpg.create_pool(
            s.dsn.get_secret_value(), min_size=s.pool_min, max_size=s.pool_max
        )

    async def stop(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def ping(self) -> bool:
        try:
            async with self.pool.acquire() as conn:
                return bool(await conn.fetchval("SELECT 1") == 1)
        except Exception:
            return False
