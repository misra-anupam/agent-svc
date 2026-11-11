from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class LimitDecision:
    allowed: bool
    remaining: int
    retry_after_seconds: float = 0.0


class RateLimiter(Protocol):
    async def acquire(self, key: str, *, rate_per_minute: int, burst: int) -> LimitDecision: ...


class TokenBudget(Protocol):
    """Per-key (IP/user) token budget over a rolling period."""

    async def check(self, key: str, *, limit: int) -> LimitDecision: ...
    async def consume(self, key: str, tokens: int, *, limit: int) -> LimitDecision: ...
