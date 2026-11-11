import httpx
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import agent_svc.adapters.registries as registries
from agent_svc.adapters.tracing.otel import OtelTracer
from agent_svc.config import Settings
from agent_svc.container import Container
from agent_svc.domain.exceptions import ConfigurationError
from agent_svc.main import create_app

BODY = {"messages": [{"role": "user", "content": "hi"}]}


async def test_request_produces_one_trace_with_run_step_and_llm_spans(
    settings: Settings, make_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    monkeypatch.setitem(
        registries.tracers._factories, "otel", (lambda s: OtelTracer(provider, "svc"), None)
    )
    settings.tracing.provider = "otel"
    async with make_client() as c:
        r = await c.post("/v1/threads/t/chat", json=BODY)
    assert r.status_code == 200
    spans = exp.get_finished_spans()
    names = {s.name for s in spans}
    assert (
        "agent.run" in names
        and "agent:respond" in names
        and any(n.startswith("llm:") for n in names)
    )
    assert len({s.context.trace_id for s in spans}) == 1  # single trace
    root = next(s for s in spans if s.name == "agent.run")
    assert root.attributes["agent.version"] == "1.0.0" and "input" not in root.attributes  # type: ignore[index]


async def test_selected_tracer_missing_config_fails_at_startup(settings: Settings) -> None:
    settings.tracing.provider = "mlflow"  # no tracking uri
    app = create_app(settings, Container(settings))
    with pytest.raises(ConfigurationError, match="MLFLOW_TRACKING_URI"):
        async with app.router.lifespan_context(app):
            pass
    settings.tracing.provider = "langfuse"  # no keys
    with pytest.raises(ConfigurationError, match="LANGFUSE_PUBLIC_KEY"):
        async with create_app(settings, Container(settings)).router.lifespan_context(app):
            pass


async def test_default_is_noop_and_requests_work(client: httpx.AsyncClient) -> None:
    assert (await client.get("/v1/agent")).status_code == 200
