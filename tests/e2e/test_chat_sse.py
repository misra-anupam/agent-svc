from collections.abc import AsyncIterator

import httpx
import pytest
from tests.helpers import parse_sse

import agent_svc.container as container_mod
from agent_svc.adapters.llm.fake import FakeLLM
from agent_svc.adapters.llm.stack import build_llm_stack
from agent_svc.config import Settings
from agent_svc.container import Container
from agent_svc.domain.exceptions import UpstreamError
from agent_svc.main import create_app


def _app_client(settings: Settings, fakes: dict[str, FakeLLM], monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        container_mod, "build_llm", lambda s: build_llm_stack(s.llm, lambda m: fakes[m])
    )
    app = create_app(settings, Container(settings))
    return app


@pytest.fixture
def chat_settings(settings: Settings) -> Settings:
    settings.orchestrator.provider = "langgraph"
    settings.llm.provider = "fake"
    settings.llm.primary_model = "fake/primary"
    settings.llm.secondary_model = "fake/secondary"
    settings.llm.max_retries = 2
    return settings


async def _post(app, body=None):  # type: ignore[no-untyped-def]
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c,
    ):
        return await c.post(
            "/v1/threads/th-1/chat",
            json=body or {"messages": [{"role": "user", "content": "hi"}]},
        )


async def test_sse_happy_path(chat_settings, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    fakes = {
        "fake/primary": FakeLLM("fake/primary", reply="the quick brown fox"),
        "fake/secondary": FakeLLM("fake/secondary"),
    }
    r = await _post(_app_client(chat_settings, fakes, monkeypatch))
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["x-run-id"] and r.headers["x-agent-version"] == "1.0.0"
    events = parse_sse(r.text)
    types = [t for t, _ in events]
    assert types[0] == "run.started" and types[-1] == "run.completed"
    assert "step.started" in types and "usage" in types
    text = "".join(d["text"] for t, d in events if t == "message.delta")  # type: ignore[misc]
    assert text == "the quick brown fox"
    assert [
        d["id"]
        for d in [
            dict(x.split(": ", 1) for x in b.splitlines() if ": " in x)
            for b in r.text.split("\n\n")
            if b.strip()
        ]
    ] == [str(i) for i in range(1, len(events) + 1)]


async def test_fallback_is_visible_in_usage_model(chat_settings, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    fakes = {
        "fake/primary": FakeLLM("fake/primary", fail_times=-1),
        "fake/secondary": FakeLLM("fake/secondary", reply="served by secondary"),
    }
    r = await _post(_app_client(chat_settings, fakes, monkeypatch))
    events = parse_sse(r.text)
    usage = next(d for t, d in events if t == "usage")
    assert usage["model"] == "fake/secondary"
    assert events[-1][0] == "run.completed"


async def test_total_llm_outage_becomes_run_error_event(chat_settings, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    err = UpstreamError("down")
    fakes = {
        "fake/primary": FakeLLM("fake/primary", fail_times=-1, error=err),
        "fake/secondary": FakeLLM("fake/secondary", fail_times=-1, error=err),
    }
    r = await _post(_app_client(chat_settings, fakes, monkeypatch))
    events = parse_sse(r.text)
    assert r.status_code == 200  # stream already started; errors travel in-band
    assert events[-1][0] == "run.error" and events[-1][1]["code"] == "upstream_error"
    assert events[-1][1]["retryable"] is True


async def test_validation_errors_are_problem_json(chat_settings, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    app = _app_client(
        chat_settings, {"fake/primary": FakeLLM(), "fake/secondary": FakeLLM()}, monkeypatch
    )
    r = await _post(app, {"messages": []})
    assert r.status_code == 422 and r.json()["code"] == "validation_error"
    r = await _post(app, {"messages": [{"role": "assistant", "content": "x"}]})
    assert r.status_code == 400 and "last message" in r.json()["detail"]
    r = await _post(app, {"messages": [{"role": "user", "content": "x" * 25_000}]})
    assert r.status_code == 413


async def test_heartbeat_flows_during_slow_runs() -> None:
    import asyncio

    from agent_svc.api.sse import sse_stream
    from agent_svc.domain.models import RunCompleted

    async def slow() -> AsyncIterator[RunCompleted]:
        await asyncio.sleep(0.15)
        yield RunCompleted(run_id="r")

    out = [b async for b in sse_stream(slow(), heartbeat_seconds=0.05)]  # type: ignore[arg-type]
    assert out[0] == b": ping\n\n" and b"run.completed" in out[-1]


async def test_client_disconnect_cancels_run() -> None:
    import asyncio

    from agent_svc.api.sse import sse_stream
    from agent_svc.domain.models import MessageDelta

    cancelled = asyncio.Event()

    async def endless() -> AsyncIterator[MessageDelta]:
        try:
            while True:
                yield MessageDelta(run_id="r", text="x")
                await asyncio.sleep(0.01)
        finally:
            cancelled.set()

    gen = sse_stream(endless(), heartbeat_seconds=1)  # type: ignore[arg-type]
    await anext(gen)
    await gen.aclose()
    await asyncio.wait_for(cancelled.wait(), 1)
