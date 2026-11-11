import asyncio
import contextlib
from collections.abc import AsyncIterator

import fakeredis

from agent_svc.adapters.cache.cached_runner import CachedRunner
from agent_svc.adapters.cache.stores import RedisCache
from agent_svc.config.settings import CacheSettings
from agent_svc.domain.exceptions import UpstreamError
from agent_svc.domain.models import (
    AgentEvent,
    AgentRequest,
    Message,
    MessageCompleted,
    MessageDelta,
    Principal,
    RunCompleted,
    RunStarted,
    Usage,
)


class CountingRunner:
    def __init__(self, fail: bool = False, delay: float = 0.0) -> None:
        self.calls, self.fail, self.delay = 0, fail, delay

    async def stream(self, request: AgentRequest) -> AsyncIterator[AgentEvent]:
        self.calls += 1
        yield RunStarted(run_id=request.run_id, thread_id="t", agent_name="a", agent_version="1")
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise UpstreamError("boom")
        text = f"answer to {request.messages[-1].content}"
        yield MessageDelta(run_id=request.run_id, text=text)
        yield MessageCompleted(run_id=request.run_id, text=text)
        yield RunCompleted(run_id=request.run_id, usage=Usage(5, 7))


def req(
    text: str = "q", user: str = "u1", tenant: str = "t1", version: str = "1", **meta: object
) -> AgentRequest:
    return AgentRequest(
        run_id=f"r-{user}-{text}",
        thread_id="th",
        principal=Principal(subject=user, tenant=tenant),
        messages=(Message("user", text),),
        agent_name="a",
        agent_version=version,
        metadata=dict(meta),
    )


async def run(r: CachedRunner, request: AgentRequest) -> list[AgentEvent]:
    return [e async for e in r.stream(request)]


def make(inner: CountingRunner, scope: str = "user") -> CachedRunner:
    return CachedRunner(
        inner, RedisCache(fakeredis.FakeAsyncRedis()), CacheSettings(provider="redis", scope=scope)
    )  # type: ignore[arg-type]


async def test_second_identical_request_is_served_from_cache() -> None:
    inner = CountingRunner()
    c = make(inner)
    first = await run(c, req())
    second = await run(c, req("  q  "))  # whitespace-insensitive
    assert inner.calls == 1
    done = second[-1]
    assert isinstance(done, RunCompleted) and done.cached and done.usage.total_tokens == 0
    assert next(e for e in second if isinstance(e, MessageCompleted)).text == "answer to q"
    assert not first[-1].cached  # type: ignore[attr-defined]


async def test_scope_user_does_not_leak_across_users() -> None:
    inner = CountingRunner()
    c = make(inner, "user")
    await run(c, req(user="alice"))
    await run(c, req(user="bob"))
    assert inner.calls == 2


async def test_scope_tenant_shares_within_tenant_only() -> None:
    inner = CountingRunner()
    c = make(inner, "tenant")
    await run(c, req(user="alice", tenant="acme"))
    await run(c, req(user="bob", tenant="acme"))
    await run(c, req(user="carol", tenant="other"))
    assert inner.calls == 2


async def test_agent_version_bump_invalidates() -> None:
    inner = CountingRunner()
    c = make(inner)
    await run(c, req(version="1"))
    await run(c, req(version="2"))
    assert inner.calls == 2


async def test_bypass_and_failures_are_not_cached() -> None:
    inner = CountingRunner()
    c = make(inner)
    await run(c, req(no_cache=True))
    await run(c, req(no_cache=True))
    assert inner.calls == 2
    bad = CountingRunner(fail=True)
    c2 = make(bad)
    for _ in range(2):
        with contextlib.suppress(UpstreamError):
            await run(c2, req())
    assert bad.calls == 2  # error answers must never be cached


async def test_single_flight_coalesces_concurrent_identical_requests() -> None:
    inner = CountingRunner(delay=0.1)
    c = make(inner)
    results = await asyncio.gather(*(run(c, req()) for _ in range(5)))
    assert inner.calls == 1
    assert sum(1 for r in results if r[-1].cached) == 4  # type: ignore[attr-defined]


async def test_cache_outage_degrades_to_miss() -> None:
    class Broken(RedisCache):
        async def get(self, key: str) -> bytes | None:
            raise ConnectionError("down")

    inner = CountingRunner()
    c = CachedRunner(inner, Broken(fakeredis.FakeAsyncRedis()), CacheSettings(provider="redis"))
    events = await run(c, req())
    assert isinstance(events[-1], RunCompleted) and inner.calls == 1
