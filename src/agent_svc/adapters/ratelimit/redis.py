"""Distributed limiters on Redis. All state transitions are single Lua scripts (atomic)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import redis.asyncio as aioredis

from agent_svc.domain.ports import LimitDecision

# Token bucket. Time comes from Redis (TIME) so app instances never disagree via clock skew.
_BUCKET = """
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local rate = tonumber(ARGV[1])      -- tokens per ms
local burst = tonumber(ARGV[2])
local d = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(d[1]) or burst
local ts = tonumber(d[2]) or now
tokens = math.min(burst, tokens + (now - ts) * rate)
local allowed, retry = 0, 0
if tokens >= 1 then tokens = tokens - 1; allowed = 1 else retry = math.ceil((1 - tokens) / rate) end
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', now)
redis.call('PEXPIRE', KEYS[1], math.ceil(burst / rate) + 1000)
return {allowed, math.floor(tokens), retry}
"""

# Concurrency lease set: ZSET of lease -> expiry(ms). Expired leases are purged on every acquire.
_ACQUIRE = """
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[1]) then return 0 end
redis.call('ZADD', KEYS[1], now + tonumber(ARGV[2]) * 1000, ARGV[3])
redis.call('PEXPIRE', KEYS[1], tonumber(ARGV[2]) * 1000 + 1000)
return 1
"""


class RedisRateLimiter:
    def __init__(self, client: aioredis.Redis, prefix: str = "rl") -> None:
        self._r, self._p = client, prefix

    async def acquire(self, key: str, *, rate_per_minute: int, burst: int) -> LimitDecision:
        res: Any = await self._r.eval(
            _BUCKET, 1, f"{self._p}:{key}", str(rate_per_minute / 60_000.0), str(burst)
        )
        allowed, remaining, retry_ms = int(res[0]), int(res[1]), int(res[2])
        return LimitDecision(bool(allowed), remaining, retry_ms / 1000.0)


class RedisTokenBudget:
    def __init__(self, client: aioredis.Redis, prefix: str = "budget") -> None:
        self._r, self._p = client, prefix

    @staticmethod
    def _day() -> str:
        return datetime.now(UTC).strftime("%Y%m%d")

    def _key(self, key: str) -> str:
        return f"{self._p}:{key}:{self._day()}"

    async def check(self, key: str, *, limit: int) -> LimitDecision:
        used = int(await self._r.get(self._key(key)) or 0)
        return LimitDecision(used < limit, max(0, limit - used), 0 if used < limit else 3600)

    async def consume(self, key: str, tokens: int, *, limit: int) -> LimitDecision:
        k = self._key(key)
        pipe = self._r.pipeline()
        pipe.incrby(k, tokens)
        pipe.expire(k, 172_800)  # keep two days so yesterday's value never lingers
        used = int((await pipe.execute())[0])
        return LimitDecision(used < limit, max(0, limit - used), 0 if used < limit else 3600)


class RedisConcurrencyGate:
    def __init__(self, client: aioredis.Redis, prefix: str = "conc") -> None:
        self._r, self._p = client, prefix

    async def acquire(self, key: str, *, limit: int, lease_seconds: int) -> str | None:
        lease = uuid.uuid4().hex
        ok: Any = await self._r.eval(
            _ACQUIRE, 1, f"{self._p}:{key}", str(limit), str(lease_seconds), lease
        )
        return lease if int(ok) == 1 else None

    async def release(self, key: str, lease: str) -> None:
        await self._r.zrem(f"{self._p}:{key}", lease)
