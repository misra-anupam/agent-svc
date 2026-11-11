from __future__ import annotations

import fakeredis
import httpx
import pytest
from pydantic import ValidationError

from agent_svc.adapters.infra.redis import RedisResource
from agent_svc.config import Settings
from agent_svc.container import Container
from agent_svc.main import create_app


async def get(settings: Settings, path: str, **kw: object) -> httpx.Response:
    fake = RedisResource(settings.redis, client=fakeredis.FakeAsyncRedis())
    app = create_app(settings, Container(settings, extra_resources=[fake]))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            return await c.get(path, **kw)  # type: ignore[arg-type]


async def test_api_responses_carry_the_hardening_headers(settings: Settings) -> None:
    h = (await get(settings, "/v1/agent")).headers
    assert h["content-security-policy"].startswith("default-src 'none'")
    assert "frame-ancestors 'none'" in h["content-security-policy"]
    assert h["x-content-type-options"] == "nosniff" and h["x-frame-options"] == "DENY"
    assert h["cache-control"] == "no-store" and h["referrer-policy"] == "no-referrer"
    assert h["cross-origin-opener-policy"] == "same-origin"
    assert "camera=()" in h["permissions-policy"]
    assert "strict-transport-security" not in h  # not production: no TLS assumption


async def test_swagger_and_demo_ui_are_not_broken_by_the_csp(settings: Settings) -> None:
    assert "content-security-policy" not in (await get(settings, "/docs")).headers
    assert "content-security-policy" not in (await get(settings, "/ui")).headers


def prod(**app: object) -> Settings:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    s.redis.enabled = False
    s.app.environment = "production"
    s.auth.enabled, s.auth.secret = True, "x" * 40  # type: ignore[assignment]
    for k, v in app.items():
        setattr(s.app, k, v)
    return s


async def test_production_turns_on_hsts_and_hides_docs_and_openapi() -> None:
    s = prod()
    r = await get(s, "/healthz")
    assert r.headers["strict-transport-security"].startswith("max-age=")
    assert (await get(s, "/docs")).status_code == 404
    assert (await get(s, "/openapi.json")).status_code == 404
    assert (await get(s, "/ui")).status_code == 404


def test_production_refuses_unsafe_configuration() -> None:
    for mutate, needle in [
        (lambda s: setattr(s.auth, "enabled", False), "auth.enabled"),
        (lambda s: setattr(s.app, "cors_origins", ["*"]), "cors_origins"),
        (lambda s: setattr(s.guardrails, "fail_mode", "open"), "fail_mode"),
    ]:
        s = prod()
        mutate(s)
        with pytest.raises(ValidationError, match=needle):
            Settings.model_validate(s.model_dump())


def test_cors_only_allows_listed_origins_and_methods() -> None:
    import asyncio

    s = prod(cors_origins=["https://app.example.com"])

    async def preflight(origin: str) -> httpx.Response:
        fake = RedisResource(s.redis, client=fakeredis.FakeAsyncRedis())
        app = create_app(s, Container(s, extra_resources=[fake]))
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c,
        ):
            return await c.options(
                "/v1/agent",
                headers={"Origin": origin, "Access-Control-Request-Method": "GET"},
            )

    ok = asyncio.run(preflight("https://app.example.com"))
    bad = asyncio.run(preflight("https://evil.example.com"))
    assert ok.headers.get("access-control-allow-origin") == "https://app.example.com"
    assert "access-control-allow-origin" not in bad.headers and bad.status_code == 400
