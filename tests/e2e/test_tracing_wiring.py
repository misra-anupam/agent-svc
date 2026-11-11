import httpx
import pytest
from tests.helpers import RecordingTracer

import agent_svc.adapters.registries as registries
from agent_svc.config import Settings
from agent_svc.container import Container
from agent_svc.domain.exceptions import ConfigurationError
from agent_svc.main import create_app

BODY = {"messages": [{"role": "user", "content": "hi"}]}


async def test_request_produces_one_trace_with_run_step_and_llm_spans(
    settings: Settings, make_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = RecordingTracer()
    monkeypatch.setitem(registries.tracers._factories, "mlflow", (lambda s: rec, None))
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


async def test_missing_tracking_uri_fails_at_startup(settings: Settings) -> None:
    settings.tracing.mlflow_tracking_uri = ""
    app = create_app(settings, Container(settings))
    with pytest.raises(ConfigurationError, match="MLFLOW_TRACKING_URI"):
        async with app.router.lifespan_context(app):
            pass


async def test_tracing_is_always_on_and_requests_work(client: httpx.AsyncClient) -> None:
    assert (await client.get("/v1/agent")).status_code == 200
