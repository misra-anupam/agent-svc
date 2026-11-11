import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from tests.helpers import RecordingTracer, make_request

from agent_svc.adapters.tracing.mlflow import MlflowTracer
from agent_svc.adapters.tracing.otel import OtelTracer
from agent_svc.adapters.tracing.traced_runner import TracedRunner
from agent_svc.config.settings import TracingSettings
from agent_svc.domain.exceptions import UpstreamError
from agent_svc.domain.models import (
    AgentEvent,
    AgentRequest,
    MessageCompleted,
    RunCompleted,
    RunError,
    RunStarted,
    StepFinished,
    StepKind,
    StepStarted,
    Usage,
    UsageReported,
)


class Scripted:
    def __init__(self, events: list[AgentEvent], boom: Exception | None = None) -> None:
        self.events, self.boom = events, boom

    async def stream(self, request: AgentRequest):  # type: ignore[no-untyped-def]
        for e in self.events:
            yield e
        if self.boom:
            raise self.boom


def events(*, fail_step: bool = False, error: bool = False) -> list[AgentEvent]:
    r = "run1"
    out: list[AgentEvent] = [
        RunStarted(run_id=r, thread_id="t1", agent_name="default", agent_version="1.0.0"),
        StepStarted(
            run_id=r, step_id="a", kind=StepKind.AGENT, name="planner", input={"q": "secret"}
        ),
        StepStarted(run_id=r, step_id="b", kind=StepKind.TOOL, name="search", parent_id="a"),
        StepFinished(
            run_id=r, step_id="b", ok=not fail_step, output_summary="top secret", duration_ms=7
        ),
        StepFinished(run_id=r, step_id="a", ok=True, duration_ms=9),
        UsageReported(run_id=r, usage=Usage(10, 4, 0.002), model="m/x"),
        MessageCompleted(run_id=r, text="the answer"),
    ]
    out.append(
        RunError(run_id=r, code="boom", message="bad")
        if error
        else RunCompleted(run_id=r, usage=Usage(10, 4))
    )
    return out


async def drain(runner: TracedRunner) -> None:
    async for _ in runner.stream(make_request()):
        pass


async def test_span_tree_nesting_usage_and_totals() -> None:
    rec = RecordingTracer()
    await drain(TracedRunner(Scripted(events()), rec, TracingSettings()))  # type: ignore[arg-type]
    by = {s.name: s for s in rec.spans}
    root = by["agent.run"]
    assert by["agent:planner"].parent is root and by["tool:search"].parent is by["agent:planner"]
    assert by["llm:m/x"].parent is root and by["llm:m/x"].attrs["gen_ai.usage.input_tokens"] == 10
    assert root.attrs["gen_ai.usage.output_tokens"] == 4 and root.attrs["cost_usd"] == 0.002
    assert root.attrs["agent.version"] == "1.0.0" and root.attrs["tenant.id"] == "default"
    assert all(s.ended for s in rec.spans) and root.error is None


async def test_content_is_not_captured_by_default_but_can_be_enabled() -> None:
    rec = RecordingTracer()
    await drain(TracedRunner(Scripted(events()), rec, TracingSettings()))  # type: ignore[arg-type]
    blob = str([s.attrs for s in rec.spans])
    assert "secret" not in blob and "the answer" not in blob and "hello world" not in blob
    rec2 = RecordingTracer()
    await drain(TracedRunner(Scripted(events()), rec2, TracingSettings(capture_content=True)))  # type: ignore[arg-type]
    blob2 = str([s.attrs for s in rec2.spans])
    assert "the answer" in blob2 and "top secret" in blob2


async def test_failures_mark_spans_as_errors() -> None:
    rec = RecordingTracer()
    await drain(TracedRunner(Scripted(events(fail_step=True, error=True)), rec, TracingSettings()))  # type: ignore[arg-type]
    by = {s.name: s for s in rec.spans}
    assert by["tool:search"].error and by["agent.run"].error == "boom: bad"
    assert by["agent.run"].attrs["error.code"] == "boom"


async def test_exceptions_and_cancellation_close_every_span() -> None:
    rec = RecordingTracer()
    evs = events()[:3]  # steps a and b started, never finished
    with pytest.raises(UpstreamError):
        await drain(TracedRunner(Scripted(evs, boom=UpstreamError("x")), rec, TracingSettings()))  # type: ignore[arg-type]
    assert all(s.ended for s in rec.spans) and rec.spans[0].error == "x"

    rec2 = RecordingTracer()

    class Hang:
        async def stream(self, request: AgentRequest):  # type: ignore[no-untyped-def]
            yield events()[0]
            await asyncio.sleep(10)

    t = asyncio.create_task(drain(TracedRunner(Hang(), rec2, TracingSettings())))  # type: ignore[arg-type]
    await asyncio.sleep(0.05)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert rec2.spans[0].ended


def test_mlflow_adapter_calls_sdk_with_types_and_parent() -> None:
    calls: list[dict[str, Any]] = []

    class FakeSpan(SimpleNamespace):
        def set_attribute(self, k: str, v: Any) -> None: ...
        def end(self) -> None:
            self.ended = True

    def start_span_no_context(**kw: Any) -> FakeSpan:
        calls.append(kw)
        return FakeSpan(trace_id="tr-1", **{"ended": False})

    tr = MlflowTracer(SimpleNamespace(start_span_no_context=start_span_no_context))
    root = tr.start_span("agent.run", attributes={"step.kind": "agent"})
    tr.start_span("tool:x", parent=root, attributes={"step.kind": "tool"})
    assert calls[0]["span_type"] == "AGENT" and calls[0]["parent_span"] is None
    assert calls[1]["span_type"] == "TOOL" and calls[1]["parent_span"] is root._span
    assert root.trace_id == "tr-1"


def _otel() -> tuple[OtelTracer, InMemorySpanExporter]:
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    return OtelTracer(provider, "svc"), exp


def test_otel_adapter_exports_parented_spans_with_error_status() -> None:
    tr, exp = _otel()
    root = tr.start_span("agent.run", attributes={"a": 1, "obj": {"x": 1}})
    child = tr.start_span("tool:x", parent=root, attributes={"step.kind": "tool"})
    child.record_error(RuntimeError("bad"))
    child.end()
    root.end()
    spans = {s.name: s for s in exp.get_finished_spans()}
    assert spans["tool:x"].parent.span_id == spans["agent.run"].context.span_id  # type: ignore[union-attr]
    assert spans["tool:x"].context.trace_id == spans["agent.run"].context.trace_id
    assert spans["tool:x"].status.status_code is StatusCode.ERROR
    assert spans["agent.run"].attributes["obj"] == "{'x': 1}" and root.trace_id == child.trace_id


def test_otel_root_span_joins_the_callers_trace_via_traceparent() -> None:
    tr, exp = _otel()
    tp = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    root = tr.start_span("agent.run", remote_parent=tp)
    root.end()
    (span,) = exp.get_finished_spans()
    assert root.trace_id == "0af7651916cd43dd8448eb211c80319c"
    assert format(span.parent.span_id, "016x") == "b7ad6b7169203331"  # type: ignore[union-attr]
