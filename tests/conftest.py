from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import fakeredis
import httpx
import pytest

import agent_svc.adapters.registries as registries
from agent_svc.adapters.infra.redis import RedisResource
from agent_svc.adapters.llm.fake import FakeLLM
from agent_svc.adapters.llm.stack import build_llm_stack
from agent_svc.config import Settings
from agent_svc.container import Container
from agent_svc.domain.models import ToolCall
from agent_svc.main import create_app
from tests.helpers import RecordingTracer


@pytest.fixture(autouse=True)
def _recording_tracer(monkeypatch: pytest.MonkeyPatch) -> None:
    """MLflow is the only tracer; tests swap in a recorder unless the tracking URI is empty."""
    from agent_svc.adapters.tracing import mlflow as mlflow_tracing

    real = mlflow_tracing.build
    monkeypatch.setitem(
        registries.tracers._factories,
        "mlflow",
        (lambda s: real(s) if not s.tracing.mlflow_tracking_uri else RecordingTracer(), None),
    )


@pytest.fixture
def settings() -> Settings:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    s.app.environment = "test"
    s.app.log_json = False
    s.redis.enabled = False  # tests inject a fake resource instead
    return s


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    fake = RedisResource(settings.redis, client=fakeredis.FakeAsyncRedis())
    fake.required = True
    app = create_app(settings, Container(settings, extra_resources=[fake]))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


@pytest.fixture
def fake_redis_server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


@pytest.fixture
def make_client(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, fake_redis_server: fakeredis.FakeServer
) -> Callable[..., object]:
    """Factory: ``async with make_client(llm_fakes, redis=True) as c`` -> httpx client.

    LLMs are injected through the real resilience stack so breaker/retry/fallback stay in play.
    """
    settings.orchestrator.provider = "langgraph"
    settings.llm.provider = "fake"
    settings.llm.primary_model = "fake/primary"
    settings.llm.secondary_model = "fake/secondary"
    settings.llm.max_retries = 2
    settings.tools.local = ["calculator"]

    @asynccontextmanager
    async def _make(
        fakes: dict[str, FakeLLM] | None = None, *, redis: bool = False
    ) -> AsyncIterator[httpx.AsyncClient]:
        llms = fakes or {
            "fake/primary": FakeLLM(
                "fake/primary",
                reply="the quick brown fox",
                tool_plan=[ToolCall("c1", "calculator", {"expression": "2+3"})],
            ),
            "fake/secondary": FakeLLM("fake/secondary"),
        }
        monkeypatch.setattr(
            registries, "build_llm", lambda s: build_llm_stack(s.llm, lambda m: llms[m])
        )
        extra = []
        if redis:
            res = RedisResource(
                settings.redis, client=fakeredis.FakeAsyncRedis(server=fake_redis_server)
            )
            res.required = True
            extra.append(res)
        app = create_app(settings, Container(settings, extra_resources=extra))
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                yield c

    return _make
