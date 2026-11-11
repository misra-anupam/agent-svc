import httpx
import pytest
from tests.helpers import RecordingTracer

import agent_svc.adapters.registries as registries
from agent_svc.config import Settings
from agent_svc.container import Container
from agent_svc.domain.exceptions import ConfigurationError
from agent_svc.main import create_app

BODY = {"messages": [{"role": "user", "content": "hi"}]}


@pytest.fixture
def rec(monkeypatch: pytest.MonkeyPatch) -> RecordingTracer:
    r = RecordingTracer()
    monkeypatch.setitem(registries.tracers._factories, "otel", (lambda s: r, None))
    return r


async def test_otel_is_the_default_tracer(settings: Settings) -> None:
    assert settings.tracing.provider == "otel"


async def test_request_produces_run_step_and_llm_spans(make_client, rec: RecordingTracer) -> None:  # type: ignore[no-untyped-def]
    async with make_client() as c:
        r = await c.post("/v1/threads/t/chat", json=BODY)
    assert r.status_code == 200
    names = {s.name for s in rec.spans}
    assert (
        "agent.run" in names
        and "tool:calculator" in names
        and any(n.startswith("llm:") for n in names)
    )
    assert all(s.ended for s in rec.spans)
    root = next(s for s in rec.spans if s.name == "agent.run")
    assert root.attrs["agent.version"] == "1.0.0" and "input" not in root.attrs


async def test_traceparent_header_becomes_the_remote_parent(
    make_client, rec: RecordingTracer
) -> None:  # type: ignore[no-untyped-def]
    tp = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    async with make_client() as c:
        await c.post("/v1/threads/t/chat", json=BODY, headers={"traceparent": tp})
        await c.post("/v1/threads/t/chat", json=BODY)
    roots = [s for s in rec.spans if s.name == "agent.run"]
    assert [s.remote_parent for s in roots] == [tp, None]


async def test_mlflow_tracer_missing_uri_fails_at_startup(settings: Settings) -> None:
    settings.tracing.provider = "mlflow"
    settings.tracing.mlflow_tracking_uri = ""
    app = create_app(settings, Container(settings))
    with pytest.raises(ConfigurationError, match="MLFLOW_TRACKING_URI"):
        async with app.router.lifespan_context(app):
            pass


async def test_requests_work_with_default_tracing(client: httpx.AsyncClient) -> None:
    assert (await client.get("/v1/agent")).status_code == 200
