import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest
from tests.helpers import parse_sse

from agent_svc.adapters.llm.fake import FakeLLM
from agent_svc.api.sse import RunSupervisor
from agent_svc.domain.exceptions import UpstreamError
from agent_svc.domain.models import MessageDelta, RunCompleted

BODY = {"messages": [{"role": "user", "content": "hi"}]}


async def test_sse_happy_path(make_client) -> None:  # type: ignore[no-untyped-def]
    async with make_client() as c:
        r = await c.post("/v1/threads/th-1/chat", json=BODY)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["x-run-id"] and r.headers["x-agent-version"] == "1.0.0"
    events = parse_sse(r.text)
    types = [t for t, _ in events]
    assert types[0] == "run.started" and types[-1] == "run.completed"
    assert "step.started" in types and "usage" in types
    assert "".join(d["text"] for t, d in events if t == "message.delta") == "the quick brown fox"  # type: ignore[misc]
    ids = [line[4:] for line in r.text.splitlines() if line.startswith("id: ")]
    assert ids == [str(i) for i in range(1, len(events) + 1)]


async def test_fallback_is_visible_in_usage_model(make_client) -> None:  # type: ignore[no-untyped-def]
    fakes = {
        "fake/primary": FakeLLM("fake/primary", fail_times=-1),
        "fake/secondary": FakeLLM("fake/secondary", reply="served by secondary"),
    }
    async with make_client(fakes) as c:
        events = parse_sse((await c.post("/v1/threads/th-1/chat", json=BODY)).text)
    assert next(d for t, d in events if t == "usage")["model"] == "fake/secondary"
    assert events[-1][0] == "run.completed"


async def test_total_llm_outage_becomes_run_error_event(make_client) -> None:  # type: ignore[no-untyped-def]
    err = UpstreamError("down")
    fakes = {
        "fake/primary": FakeLLM("fake/primary", fail_times=-1, error=err),
        "fake/secondary": FakeLLM("fake/secondary", fail_times=-1, error=err),
    }
    async with make_client(fakes) as c:
        r = await c.post("/v1/threads/th-1/chat", json=BODY)
    events = parse_sse(r.text)
    assert r.status_code == 200  # stream already started; errors travel in-band
    assert events[-1][0] == "run.error" and events[-1][1]["code"] == "upstream_error"
    assert events[-1][1]["retryable"] is True


async def test_validation_errors_are_problem_json(make_client) -> None:  # type: ignore[no-untyped-def]
    async with make_client() as c:
        r = await c.post("/v1/threads/th-1/chat", json={"messages": []})
        assert r.status_code == 422 and r.json()["code"] == "validation_error"
        r = await c.post(
            "/v1/threads/th-1/chat", json={"messages": [{"role": "assistant", "content": "x"}]}
        )
        assert r.status_code == 400 and "last message" in r.json()["detail"]
        r = await c.post(
            "/v1/threads/th-1/chat", json={"messages": [{"role": "user", "content": "x" * 25_000}]}
        )
        assert r.status_code == 413
        r = await c.post("/v1/threads/bad%20id/chat", json=BODY)
        assert r.status_code == 422


async def _slow_run() -> AsyncIterator[RunCompleted]:
    await asyncio.sleep(0.15)
    yield RunCompleted(run_id="r")


async def test_heartbeat_flows_during_slow_runs() -> None:
    from agent_svc.adapters.runs.buffer import InMemoryRunBuffer

    sup = RunSupervisor(InMemoryRunBuffer(), detach_on_disconnect=False)
    await sup.start("r1", "u", _slow_run())  # type: ignore[arg-type]
    out = [b async for b in sup.tail("r1", heartbeat_seconds=0.05)]
    assert out[0] == b": ping\n\n" and b"run.completed" in out[-1]


async def test_disconnect_cancels_run_unless_detached() -> None:
    from agent_svc.adapters.runs.buffer import InMemoryRunBuffer

    for detach in (False, True):
        cancelled = asyncio.Event()
        finished = asyncio.Event()

        async def endless(
            cancelled: asyncio.Event = cancelled, finished: asyncio.Event = finished
        ) -> AsyncIterator[MessageDelta]:
            try:
                for _ in range(8):
                    yield MessageDelta(run_id="r", text="x")
                    await asyncio.sleep(0.02)
                finished.set()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        sup = RunSupervisor(InMemoryRunBuffer(), detach_on_disconnect=detach)
        await sup.start("r", "u", endless())  # type: ignore[arg-type]
        gen = sup.tail("r", heartbeat_seconds=1)
        await anext(gen)
        await gen.aclose()  # client goes away
        if detach:
            await asyncio.wait_for(finished.wait(), 2)
            assert not cancelled.is_set()
        else:
            await asyncio.wait_for(cancelled.wait(), 2)


@pytest.mark.parametrize("path", ["/v1/agent"])
async def test_versioned_routes_present(client: httpx.AsyncClient, path: str) -> None:
    assert (await client.get(path)).status_code == 200
