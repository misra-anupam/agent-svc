"""TracedRunner: backend-neutral tracing decorator for any AgentRunner.

Span tree: ``agent.run`` (root) -> one child per step (tool/rag/agent/mcp/guardrail), nested via
``parent_id`` -> one ``llm`` child per usage report (model, tokens, cost).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import structlog

from agent_svc.config.settings import TracingSettings
from agent_svc.domain.models import (
    AgentEvent,
    AgentRequest,
    MessageCompleted,
    RunCompleted,
    RunError,
    StepFinished,
    StepStarted,
    UsageReported,
)
from agent_svc.domain.ports import AgentRunner, Span, Tracer


class TracedRunner:
    def __init__(self, inner: AgentRunner, tracer: Tracer, settings: TracingSettings) -> None:
        self._inner, self._tracer, self._capture = inner, tracer, settings.capture_content

    async def stream(self, request: AgentRequest) -> AsyncIterator[AgentEvent]:
        p = request.principal
        root = self._tracer.start_span(
            "agent.run",
            attributes={
                "step.kind": "agent",
                "agent.name": request.agent_name,
                "agent.version": request.agent_version,
                "run.id": request.run_id,
                "thread.id": request.thread_id,
                "enduser.id": p.subject,
                "tenant.id": p.tenant,
                "user.tier": p.tier,
                **({"input": request.last_user_text} if self._capture else {}),
            },
            remote_parent=request.metadata.get("traceparent"),
        )
        if root.trace_id:
            structlog.contextvars.bind_contextvars(trace_id=root.trace_id)
        open_spans: dict[str, Span] = {}
        tokens_in = tokens_out = 0
        cost = 0.0
        try:
            async for ev in self._inner.stream(request):
                if isinstance(ev, StepStarted):
                    attrs: dict[str, object] = {"step.kind": ev.kind.value, "step.name": ev.name}
                    if self._capture and ev.input:
                        attrs["input"] = str(ev.input)
                    open_spans[ev.step_id] = self._tracer.start_span(
                        f"{ev.kind.value}:{ev.name}",
                        parent=open_spans.get(ev.parent_id or "", root),
                        attributes=attrs,
                    )
                elif isinstance(ev, StepFinished):
                    span = open_spans.pop(ev.step_id, None)
                    if span:
                        span.set_attribute("step.duration_ms", ev.duration_ms)
                        if self._capture:
                            span.set_attribute("output", ev.output_summary)
                        if not ev.ok:
                            span.record_error(RuntimeError(ev.output_summary or "step failed"))
                        span.end()
                elif isinstance(ev, UsageReported):
                    u = ev.usage
                    tokens_in += u.input_tokens
                    tokens_out += u.output_tokens
                    cost += u.cost_usd
                    llm = self._tracer.start_span(
                        f"llm:{ev.model or 'unknown'}",
                        parent=root,
                        attributes={
                            "step.kind": "llm",
                            "gen_ai.request.model": ev.model,
                            "gen_ai.usage.input_tokens": u.input_tokens,
                            "gen_ai.usage.output_tokens": u.output_tokens,
                            "cost_usd": u.cost_usd,
                        },
                    )
                    llm.end()
                elif isinstance(ev, MessageCompleted) and self._capture:
                    root.set_attribute("output", ev.text)
                elif isinstance(ev, RunError):
                    root.set_attribute("error.code", ev.code)
                    root.record_error(RuntimeError(f"{ev.code}: {ev.message}"))
                elif isinstance(ev, RunCompleted):
                    root.set_attribute("gen_ai.usage.input_tokens", tokens_in)
                    root.set_attribute("gen_ai.usage.output_tokens", tokens_out)
                    root.set_attribute("cost_usd", cost)
                    root.set_attribute("cache.hit", ev.cached)
                yield ev
        except BaseException as exc:  # includes CancelledError: mark and re-raise
            root.record_error(exc)
            raise
        finally:
            for span in open_spans.values():  # steps that never finished (error/cancel)
                span.end()
            root.end()
