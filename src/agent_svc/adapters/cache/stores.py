from __future__ import annotations

import redis.asyncio as aioredis


class RedisCache:
    def __init__(self, client: aioredis.Redis, prefix: str = "cache") -> None:
        self._r, self._p = client, prefix

    async def get(self, key: str) -> bytes | None:
        val = await self._r.get(f"{self._p}:{key}")
        return val if isinstance(val, bytes) else str(val).encode() if val is not None else None

    async def set(self, key: str, value: bytes, ttl_seconds: int) -> None:
        await self._r.set(f"{self._p}:{key}", value, ex=ttl_seconds)

    async def set_if_absent(self, key: str, value: bytes, ttl_seconds: int) -> bool:
        return bool(await self._r.set(f"{self._p}:{key}", value, ex=ttl_seconds, nx=True))

    async def delete(self, key: str) -> None:
        await self._r.delete(f"{self._p}:{key}")
