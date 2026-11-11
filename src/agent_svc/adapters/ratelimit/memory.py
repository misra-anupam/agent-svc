"""In-process implementations (single instance / tests)."""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from agent_svc.domain.ports import LimitDecision


class InMemoryRateLimiter:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}

    async def acquire(self, key: str, *, rate_per_minute: int, burst: int) -> LimitDecision:
        now, rate = self._clock(), rate_per_minute / 60.0
        tokens, ts = self._buckets.get(key, (float(burst), now))
        tokens = min(float(burst), tokens + (now - ts) * rate)
        if tokens >= 1:
            self._buckets[key] = (tokens - 1, now)
            return LimitDecision(True, int(tokens - 1))
        self._buckets[key] = (tokens, now)
        return LimitDecision(False, 0, (1 - tokens) / rate)


def _day() -> str:
    return datetime.now(UTC).strftime("%Y%m%d")


class InMemoryTokenBudget:
    def __init__(self) -> None:
        self._used: dict[str, int] = {}

    async def check(self, key: str, *, limit: int) -> LimitDecision:
        used = self._used.get(f"{key}:{_day()}", 0)
        return LimitDecision(used < limit, max(0, limit - used), 0 if used < limit else 3600)

    async def consume(self, key: str, tokens: int, *, limit: int) -> LimitDecision:
        k = f"{key}:{_day()}"
        self._used[k] = self._used.get(k, 0) + tokens
        return await self.check(key, limit=limit)


class InMemoryConcurrencyGate:
    def __init__(self) -> None:
        self._leases: dict[str, dict[str, float]] = {}

    async def acquire(self, key: str, *, limit: int, lease_seconds: int) -> str | None:
        now = time.monotonic()
        held = {k: exp for k, exp in self._leases.get(key, {}).items() if exp > now}
        if len(held) >= limit:
            self._leases[key] = held
            return None
        lease = uuid.uuid4().hex
        held[lease] = now + lease_seconds
        self._leases[key] = held
        return lease

    async def release(self, key: str, lease: str) -> None:
        self._leases.get(key, {}).pop(lease, None)


def seconds_ceil(x: float) -> int:
    return max(1, math.ceil(x))
