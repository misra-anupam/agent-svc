import httpx
import pytest

from agent_svc.adapters.infra.redis import RedisResource
from agent_svc.config import Settings
from agent_svc.container import Container
from agent_svc.main import create_app


async def test_liveness_and_headers(client: httpx.AsyncClient) -> None:
    r = await client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    assert r.headers["x-request-id"] and r.headers["x-agent-version"] == "1.0.0"
    assert r.headers["x-content-type-options"] == "nosniff"


async def test_request_id_is_propagated_when_valid(client: httpx.AsyncClient) -> None:
    r = await client.get("/healthz", headers={"X-Request-ID": "abc-12345678"})
    assert r.headers["x-request-id"] == "abc-12345678"
    r = await client.get("/healthz", headers={"X-Request-ID": "bad id\r\n"})
    assert r.headers["x-request-id"] != "bad id"


async def test_readiness_reports_checks(client: httpx.AsyncClient) -> None:
    r = await client.get("/readyz")
    assert r.status_code == 200
    assert r.json() == {"status": "ready", "checks": {"redis": True}}


async def test_agent_info_versioned_route(client: httpx.AsyncClient) -> None:
    r = await client.get("/v1/agent")
    assert r.json() == {"name": "default", "version": "1.0.0", "api_version": "v1"}


async def test_unknown_route_is_problem_json(client: httpx.AsyncClient) -> None:
    r = await client.get("/nope")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/problem+json")
    assert r.json()["code"] == "http_404" and r.json()["request_id"]


async def test_metrics_exposed(client: httpx.AsyncClient) -> None:
    await client.get("/healthz")
    r = await client.get("/metrics/")
    assert "agent_http_requests_total" in r.text


async def test_readiness_503_when_required_dependency_down(settings: Settings) -> None:
    class Down(RedisResource):
        async def ping(self) -> bool:
            return False

    dead = Down(settings.redis, client=object())  # type: ignore[arg-type]
    dead.required = True
    dead.stop = lambda: _noop()  # type: ignore[method-assign]
    app = create_app(settings, Container(settings, extra_resources=[dead]))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c,
    ):
        r = await c.get("/readyz")
    assert r.status_code == 503 and r.json()["status"] == "unavailable"


async def _noop() -> None:
    return None


@pytest.mark.parametrize("provider", ["adk"])
async def test_missing_extra_fails_startup_clearly(
    settings: Settings, provider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_svc.adapters import registries as reg
    from agent_svc.domain.exceptions import ConfigurationError

    # every extra is installed in dev, so simulate the SDK being absent
    monkeypatch.setitem(
        reg.orchestrators._factories,
        provider,
        ("agent_svc.adapters.orchestrators._absent_sdk:build", provider),
    )
    settings.orchestrator.provider = provider  # type: ignore[assignment]
    app = create_app(settings)
    with pytest.raises(ConfigurationError, match="uv sync --extra adk"):
        async with app.router.lifespan_context(app):
            pass


async def test_demo_ui_served_outside_production(client: httpx.AsyncClient) -> None:
    r = await client.get("/ui")
    assert r.status_code == 200 and "agent-svc demo" in r.text
