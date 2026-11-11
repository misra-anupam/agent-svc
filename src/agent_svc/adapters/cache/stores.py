from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable

import redis.asyncio as aioredis


class InMemoryCache:
    """TTL + LRU cache for single-instance use and tests."""

    def __init__(
        self, max_entries: int = 1000, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._d: OrderedDict[str, tuple[float, bytes]] = OrderedDict()
        self._max, self._clock = max_entries, clock

    async def get(self, key: str) -> bytes | None:
        hit = self._d.get(key)
        if hit is None:
            return None
        if hit[0] <= self._clock():
            del self._d[key]
            return None
        self._d.move_to_end(key)
        return hit[1]

    async def set(self, key: str, value: bytes, ttl_seconds: int) -> None:
        self._d[key] = (self._clock() + ttl_seconds, value)
        self._d.move_to_end(key)
        while len(self._d) > self._max:
            self._d.popitem(last=False)

    async def set_if_absent(self, key: str, value: bytes, ttl_seconds: int) -> bool:
        if await self.get(key) is not None:
            return False
        await self.set(key, value, ttl_seconds)
        return True

    async def delete(self, key: str) -> None:
        self._d.pop(key, None)


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
