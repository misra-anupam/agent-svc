from __future__ import annotations

import redis.asyncio as aioredis

from agent_svc.config.settings import RedisSettings


class RedisResource:
    """Shared async Redis client; implements Resource and HealthCheck."""

    name = "redis"

    def __init__(self, settings: RedisSettings, client: aioredis.Redis | None = None) -> None:
        self._settings = settings
        self.required = settings.required
        self._client = client

    @property
    def client(self) -> aioredis.Redis:
        if self._client is None:
            raise RuntimeError("RedisResource not started")
        return self._client

    async def start(self) -> None:
        if self._client is None:
            self._client = aioredis.from_url(self._settings.url, decode_responses=False)

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def ping(self) -> bool:
        try:
            return bool(await self.client.ping())
        except Exception:
            return False
