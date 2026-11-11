import time

import pytest

from agent_svc.adapters.ratelimit.memory import (
    InMemoryConcurrencyGate,
    InMemoryRateLimiter,
    InMemoryTokenBudget,
)
from agent_svc.application.admission import AdmissionService
from agent_svc.config.settings import RateLimitSettings, TierLimit
from agent_svc.domain.exceptions import BudgetExceededError, RateLimitedError, UnavailableError
from agent_svc.domain.models import Principal, Usage


def svc(**kw: object) -> AdmissionService:
    s = RateLimitSettings(
        provider="inmemory",
        tiers={
            "free": TierLimit(
                requests_per_minute=60, burst=2, daily_token_budget=100, max_concurrent=1
            ),
            "solo": TierLimit(
                requests_per_minute=6000, burst=50, daily_token_budget=10_000, max_concurrent=1
            ),
            "pro": TierLimit(
                requests_per_minute=6000, burst=50, daily_token_budget=10_000, max_concurrent=5
            ),
        },
        ip_daily_token_budget=150,
        **kw,  # type: ignore[arg-type]
    )
    return AdmissionService(
        s,
        limiter=InMemoryRateLimiter(),
        budget=InMemoryTokenBudget(),
        gate=InMemoryConcurrencyGate(),
    )


def user(tier: str = "free", sub: str = "u1", ip: str = "1.1.1.1") -> Principal:
    return Principal(subject=sub, tier=tier, ip=ip)


async def test_reject_mode_returns_retry_after() -> None:
    s = svc()
    for _ in range(2):
        await s.release(await s.admit(user()))
    with pytest.raises(RateLimitedError) as ei:
        await s.admit(user())
    assert ei.value.retry_after >= 1 and ei.value.status_code == 429


async def test_throttle_mode_delays_instead_of_rejecting() -> None:
    s = svc(mode="throttle", max_throttle_delay_ms=2000)
    for _ in range(2):
        await s.release(await s.admit(user()))
    t0 = time.monotonic()
    adm = await s.admit(user())  # bucket empty: waits ~1s for a token, then succeeds
    assert time.monotonic() - t0 >= 0.5
    await s.release(adm)


async def test_tiers_have_different_limits() -> None:
    s = svc()
    for _ in range(10):
        await s.release(await s.admit(user("pro", "p1")))  # pro burst is 50
    for _ in range(2):
        await s.release(await s.admit(user("free", "f1")))
    with pytest.raises(RateLimitedError):
        await s.admit(user("free", "f1"))


async def test_unknown_tier_claim_falls_back_to_default() -> None:
    s = svc()
    name, tier = s.tier_for(user("platinum-hacker"))
    assert name == "free" and tier.burst == 2


async def test_token_budget_per_user_and_per_ip() -> None:
    s = svc()
    p = user("pro", "p1", ip="8.8.8.8")
    await s.record_usage(p, Usage(100, 60))  # 160 tokens > IP budget (150), < pro budget
    with pytest.raises(BudgetExceededError) as ei:
        await s.admit(p)
    assert ei.value.code == "token_budget_exceeded" and ei.value.status_code == 429
    await s.admit(user("pro", "p2", ip="9.9.9.9"))  # different user + IP unaffected
    # another user behind the same IP is blocked too (IP-level budget)
    with pytest.raises(BudgetExceededError):
        await s.admit(user("pro", "someone-else", ip="8.8.8.8"))


async def test_concurrency_limit_and_release() -> None:
    s = svc()
    first = await s.admit(user("solo"))
    with pytest.raises(RateLimitedError, match="concurrent"):
        await s.admit(user("solo"))
    await s.release(first)
    await s.release(await s.admit(user("solo")))


class _Boom:
    async def acquire(self, *a: object, **k: object) -> None:
        raise ConnectionError("redis down")


async def test_backend_failure_fail_open_vs_closed() -> None:
    open_ = AdmissionService(
        RateLimitSettings(fail_open=True), limiter=_Boom(), budget=None, gate=None
    )  # type: ignore[arg-type]
    await open_.admit(user())
    closed = AdmissionService(
        RateLimitSettings(fail_open=False), limiter=_Boom(), budget=None, gate=None
    )  # type: ignore[arg-type]
    with pytest.raises(UnavailableError):
        await closed.admit(user())
