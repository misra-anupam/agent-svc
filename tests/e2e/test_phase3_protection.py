import asyncio
import time

import jwt
import pytest
from pydantic import SecretStr
from tests.helpers import parse_sse

from agent_svc.adapters.llm.fake import FakeLLM
from agent_svc.config.settings import TierLimit

SECRET = "s3cret-s3cret-s3cret-s3cret-123456"
BODY = {"messages": [{"role": "user", "content": "hi"}]}


def bearer(sub: str = "alice", **claims: object) -> dict[str, str]:
    t = jwt.encode({"sub": sub, "exp": int(time.time()) + 300, **claims}, SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {t}"}


@pytest.fixture
def jwt_on(settings):  # type: ignore[no-untyped-def]
    settings.auth.enabled = True
    settings.auth.secret = SecretStr(SECRET)
    return settings


async def test_auth_on_rejects_missing_and_bad_tokens_with_challenge(jwt_on, make_client) -> None:  # type: ignore[no-untyped-def]
    async with make_client() as c:
        r = await c.post("/v1/threads/t1/chat", json=BODY)
        assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
        assert r.headers["content-type"].startswith("application/problem+json")
        r = await c.post("/v1/threads/t1/chat", json=BODY, headers={"Authorization": "Bearer junk"})
        assert r.status_code == 401 and "junk" not in r.text
        r = await c.post("/v1/threads/t1/chat", json=BODY, headers=bearer())
        assert r.status_code == 200
        assert (await c.get("/healthz")).status_code == 200  # ops endpoints stay open


async def test_auth_off_allows_anonymous(make_client) -> None:  # type: ignore[no-untyped-def]
    async with make_client() as c:
        assert (await c.post("/v1/threads/t1/chat", json=BODY)).status_code == 200


async def test_rate_limit_returns_429_with_retry_after(settings, make_client) -> None:  # type: ignore[no-untyped-def]
    settings.ratelimit.provider = "inmemory"
    settings.ratelimit.tiers["free"] = TierLimit(
        requests_per_minute=60, burst=2, daily_token_budget=10**9, max_concurrent=10
    )
    async with make_client() as c:
        codes = [(await c.post("/v1/threads/t1/chat", json=BODY)).status_code for _ in range(3)]
        assert codes == [200, 200, 429]
        r = await c.post("/v1/threads/t1/chat", json=BODY)
        assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
        assert r.json()["code"] == "rate_limited"


async def test_tier_from_jwt_changes_limit(jwt_on, settings, make_client) -> None:  # type: ignore[no-untyped-def]
    settings.ratelimit.provider = "redis"
    settings.ratelimit.tiers["free"] = TierLimit(
        requests_per_minute=60, burst=1, daily_token_budget=10**9, max_concurrent=10
    )
    settings.ratelimit.tiers["pro"] = TierLimit(
        requests_per_minute=60, burst=5, daily_token_budget=10**9, max_concurrent=10
    )
    async with make_client(redis=True) as c:
        free = [
            (await c.post("/v1/threads/t/chat", json=BODY, headers=bearer("f"))).status_code
            for _ in range(3)
        ]
        pro = [
            (
                await c.post("/v1/threads/t/chat", json=BODY, headers=bearer("p", tier="pro"))
            ).status_code
            for _ in range(3)
        ]
    assert free == [200, 429, 429] and pro == [200, 200, 200]


async def test_token_budget_is_enforced_from_real_usage(settings, make_client) -> None:  # type: ignore[no-untyped-def]
    settings.ratelimit.provider = "redis"
    settings.ratelimit.ip_daily_token_budget = 3  # FakeLLM reports ~5 tokens for one run
    async with make_client(redis=True) as c:
        assert (await c.post("/v1/threads/t/chat", json=BODY)).status_code == 200
        r = await c.post("/v1/threads/t/chat", json=BODY)
    assert r.status_code == 429 and r.json()["code"] == "token_budget_exceeded"


async def test_concurrency_cap_rejects_second_parallel_run(settings, make_client) -> None:  # type: ignore[no-untyped-def]
    settings.ratelimit.provider = "inmemory"
    settings.ratelimit.tiers["free"] = TierLimit(
        requests_per_minute=6000, burst=50, daily_token_budget=10**9, max_concurrent=1
    )
    slow = {
        "fake/primary": FakeLLM("fake/primary", reply="a b c d e", delay=0.1),
        "fake/secondary": FakeLLM("fake/secondary"),
    }
    async with make_client(slow) as c:
        r1, r2 = await asyncio.gather(c.post("/v1/threads/t/chat", json=BODY), _later(c))
        assert r1.status_code == 200 and r2.status_code == 429
        assert (await c.post("/v1/threads/t/chat", json=BODY)).status_code == 200  # slot freed


async def _later(c):  # type: ignore[no-untyped-def]
    await asyncio.sleep(0.05)
    return await c.post("/v1/threads/t/chat", json=BODY)


async def test_answer_cache_serves_second_request(settings, make_client) -> None:  # type: ignore[no-untyped-def]
    settings.cache.provider = "inmemory"
    primary = FakeLLM("fake/primary", reply="cached answer here")
    async with make_client({"fake/primary": primary, "fake/secondary": FakeLLM()}) as c:
        a = parse_sse((await c.post("/v1/threads/t1/chat", json=BODY)).text)
        b = parse_sse((await c.post("/v1/threads/t2/chat", json=BODY)).text)
        d = parse_sse(
            (
                await c.post(
                    "/v1/threads/t3/chat", json=BODY, headers={"Cache-Control": "no-cache"}
                )
            ).text
        )
    assert primary.calls == 2  # 1st miss, 2nd hit, 3rd bypass
    assert (
        a[-1][1]["cached"] is False and b[-1][1]["cached"] is True and d[-1][1]["cached"] is False
    )


async def test_stateful_memory_and_thread_isolation(jwt_on, settings, make_client) -> None:  # type: ignore[no-untyped-def]
    settings.memory.thread_provider = "redis"
    settings.memory.user_provider = "inmemory"
    primary = FakeLLM("fake/primary", reply="noted")
    async with make_client({"fake/primary": primary, "fake/secondary": FakeLLM()}, redis=True) as c:
        alice = bearer("alice")
        await c.post(
            "/v1/threads/th/chat",
            json={"messages": [{"role": "user", "content": "my name is Alice"}]},
            headers=alice,
        )
        await c.post(
            "/v1/threads/th/chat",
            json={"messages": [{"role": "user", "content": "what is my name"}]},
            headers=alice,
        )
        seen = [m.content for m in primary.seen[-1]]
        assert seen == ["my name is Alice", "noted", "what is my name"]  # server stitched history

        hist = await c.get("/v1/threads/th", headers=alice)
        assert hist.status_code == 200 and len(hist.json()["messages"]) == 4
        assert (await c.get("/v1/threads/th", headers=bearer("mallory"))).status_code == 404  # IDOR

        await c.put("/v1/memory/user", json={"facts": {"lang": "en"}}, headers=alice)
        await c.post("/v1/threads/th2/chat", json=BODY, headers=alice)
        assert primary.seen[-1][0].role == "system" and "lang=en" in primary.seen[-1][0].content

        assert (await c.delete("/v1/threads/th", headers=alice)).status_code == 204
        assert (await c.get("/v1/threads/th", headers=alice)).status_code == 404
        assert (await c.delete("/v1/memory", headers=alice)).status_code == 204
        assert (await c.get("/v1/memory/user", headers=alice)).json() == {"facts": {}}


async def test_idempotency_key_replays_instead_of_rerunning(make_client) -> None:  # type: ignore[no-untyped-def]
    primary = FakeLLM("fake/primary", reply="only once")
    h = {"Idempotency-Key": "key-12345678"}
    async with make_client({"fake/primary": primary, "fake/secondary": FakeLLM()}) as c:
        r1 = await c.post("/v1/threads/t/chat", json=BODY, headers=h)
        r2 = await c.post("/v1/threads/t/chat", json=BODY, headers=h)
        r3 = await c.post(
            "/v1/threads/other/chat", json=BODY, headers=h
        )  # different thread = new run
    assert primary.calls == 2
    assert r1.headers["x-run-id"] == r2.headers["x-run-id"] != r3.headers["x-run-id"]
    assert parse_sse(r1.text) == parse_sse(r2.text)


async def test_resume_from_last_event_id_and_owner_check(jwt_on, make_client) -> None:  # type: ignore[no-untyped-def]
    async with make_client() as c:
        r = await c.post("/v1/threads/t/chat", json=BODY, headers=bearer("alice"))
        run_id, total = r.headers["x-run-id"], len(parse_sse(r.text))
        tail = await c.get(
            f"/v1/runs/{run_id}/events", headers={**bearer("alice"), "Last-Event-ID": "3"}
        )
        events = parse_sse(tail.text)
        assert len(events) == total - 3 and events[-1][0] == "run.completed"
        assert "id: 4\n" in tail.text
        full = await c.get(f"/v1/runs/{run_id}/events", headers=bearer("alice"))
        assert len(parse_sse(full.text)) == total
        assert (
            await c.get(f"/v1/runs/{run_id}/events", headers=bearer("mallory"))
        ).status_code == 404
        assert (
            await c.get("/v1/runs/doesnotexist/events", headers=bearer("alice"))
        ).status_code == 404
