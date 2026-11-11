"""Contract tests: every backend of a port must behave identically (in-memory vs Redis)."""

import asyncio

import fakeredis
import pytest

from agent_svc.adapters.cache.stores import InMemoryCache, RedisCache
from agent_svc.adapters.ratelimit.memory import (
    InMemoryConcurrencyGate,
    InMemoryRateLimiter,
    InMemoryTokenBudget,
)
from agent_svc.adapters.ratelimit.redis import (
    RedisConcurrencyGate,
    RedisRateLimiter,
    RedisTokenBudget,
)
from agent_svc.adapters.runs.buffer import InMemoryRunBuffer, RedisRunBuffer


@pytest.fixture(params=["inmemory", "redis"])
def backend(request: pytest.FixtureRequest):  # type: ignore[no-untyped-def]
    r = fakeredis.FakeAsyncRedis()
    if request.param == "inmemory":
        return (
            InMemoryRateLimiter(),
            InMemoryTokenBudget(),
            InMemoryConcurrencyGate(),
            InMemoryCache(),
            InMemoryRunBuffer(),
        )
    return (
        RedisRateLimiter(r),
        RedisTokenBudget(r),
        RedisConcurrencyGate(r),
        RedisCache(r),
        RedisRunBuffer(r),
    )


async def test_token_bucket_burst_then_deny_then_refill(backend) -> None:  # type: ignore[no-untyped-def]
    limiter = backend[0]
    kw = {"rate_per_minute": 6000, "burst": 2}  # 100 tokens/second
    assert (await limiter.acquire("k", **kw)).allowed
    assert (await limiter.acquire("k", **kw)).allowed
    denied = await limiter.acquire("k", **kw)
    assert not denied.allowed and 0 < denied.retry_after_seconds <= 0.02
    await asyncio.sleep(0.03)
    assert (await limiter.acquire("k", **kw)).allowed
    assert (await limiter.acquire("other", **kw)).allowed  # keys are independent


async def test_token_budget_blocks_after_limit(backend) -> None:  # type: ignore[no-untyped-def]
    budget = backend[1]
    assert (await budget.check("ip:1", limit=100)).allowed
    assert (await budget.consume("ip:1", 60, limit=100)).remaining == 40
    assert (await budget.consume("ip:1", 60, limit=100)).allowed is False
    assert (await budget.check("ip:1", limit=100)).allowed is False
    assert (await budget.check("ip:2", limit=100)).allowed


async def test_concurrency_gate_limits_and_releases(backend) -> None:  # type: ignore[no-untyped-def]
    gate = backend[2]
    a = await gate.acquire("u", limit=2, lease_seconds=30)
    b = await gate.acquire("u", limit=2, lease_seconds=30)
    assert a and b and await gate.acquire("u", limit=2, lease_seconds=30) is None
    await gate.release("u", a)
    assert await gate.acquire("u", limit=2, lease_seconds=30)


async def test_concurrency_leases_expire(backend) -> None:  # type: ignore[no-untyped-def]
    gate = backend[2]
    assert await gate.acquire("crashy", limit=1, lease_seconds=1)
    assert await gate.acquire("crashy", limit=1, lease_seconds=1) is None
    await asyncio.sleep(1.1)  # a crashed worker never released: the lease must lapse
    assert await gate.acquire("crashy", limit=1, lease_seconds=1)


async def test_cache_roundtrip_ttl_and_set_if_absent(backend) -> None:  # type: ignore[no-untyped-def]
    cache = backend[3]
    assert await cache.get("a") is None
    await cache.set("a", b"1", 5)
    assert await cache.get("a") == b"1"
    assert await cache.set_if_absent("a", b"2", 5) is False
    assert await cache.set_if_absent("b", b"2", 5) is True
    await cache.delete("a")
    assert await cache.get("a") is None
    await cache.set("short", b"x", 1)
    await asyncio.sleep(1.1)
    assert await cache.get("short") is None


async def test_run_buffer_append_read_finish_owner(backend) -> None:  # type: ignore[no-untyped-def]
    buf = backend[4]
    await buf.start("r1", "alice")
    assert await buf.owner("r1") == "alice" and await buf.owner("nope") is None
    await buf.append("r1", b"f1")
    await buf.append("r1", b"f2")
    frames, done = await buf.read("r1", 0, 0.05)
    assert frames == [b"f1", b"f2"] and not done
    assert await buf.read("r1", 1, 0.05) == ([b"f2"], False)
    assert await buf.read("r1", 2, 0.05) == ([], False)  # nothing new yet, still running
    await buf.finish("r1")
    assert await buf.read("r1", 2, 0.05) == ([], True)


async def test_run_buffer_read_wakes_on_new_frame(backend) -> None:  # type: ignore[no-untyped-def]
    buf = backend[4]
    await buf.start("r2", "u")

    async def writer() -> None:
        await asyncio.sleep(0.05)
        await buf.append("r2", b"late")

    t = asyncio.create_task(writer())
    frames, _ = await buf.read("r2", 0, 2.0)
    await t
    assert frames == [b"late"]
