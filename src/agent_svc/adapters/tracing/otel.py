"""OpenTelemetry tracer (OTLP/HTTP): Jaeger, Tempo, Datadog, Honeycomb, any OTLP collector."""

from __future__ import annotations

import asyncio
from typing import Any

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace import Span as OtelSpanApi
from opentelemetry.trace import Status, StatusCode, set_span_in_context
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from agent_svc import __version__
from agent_svc.config.settings import Settings

_PRIMITIVES = (str, bool, int, float)


def _clean(attrs: dict[str, Any] | None) -> dict[str, Any]:
    return {k: v if isinstance(v, _PRIMITIVES) else str(v) for k, v in (attrs or {}).items()}


class OtelSpan:
    def __init__(self, span: OtelSpanApi) -> None:
        self._span = span
        self.trace_id = format(span.get_span_context().trace_id, "032x")

    def set_attribute(self, key: str, value: Any) -> None:
        self._span.set_attribute(key, value if isinstance(value, _PRIMITIVES) else str(value))

    def record_error(self, exc: BaseException) -> None:
        self._span.record_exception(exc)
        self._span.set_status(Status(StatusCode.ERROR, str(exc)))

    def end(self) -> None:
        self._span.end()


class OtelTracer:
    def __init__(self, provider: TracerProvider, service_name: str) -> None:
        self._provider = provider
        self._tracer = provider.get_tracer(service_name, __version__)

    def start_span(
        self,
        name: str,
        *,
        parent: Any = None,
        attributes: dict[str, Any] | None = None,
        remote_parent: str | None = None,
    ) -> OtelSpan:
        ctx = None
        if isinstance(parent, OtelSpan):
            ctx = set_span_in_context(parent._span)
        elif remote_parent:  # W3C ``traceparent`` of the caller: join its distributed trace
            ctx = TraceContextTextMapPropagator().extract({"traceparent": remote_parent})
        return OtelSpan(self._tracer.start_span(name, context=ctx, attributes=_clean(attributes)))

    async def flush(self) -> None:
        await asyncio.to_thread(self._provider.force_flush, 5000)

    async def shutdown(self) -> None:
        await asyncio.to_thread(self._provider.shutdown)


def build(settings: Settings) -> OtelTracer:
    t = settings.tracing
    provider = TracerProvider(
        resource=Resource.create({"service.name": t.service_name, "service.version": __version__}),
        sampler=ParentBased(TraceIdRatioBased(t.sample_ratio)),
    )
    if t.otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        endpoint = t.otlp_endpoint.rstrip("/") + "/v1/traces"
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=t.otlp_headers or None))
        )
    return OtelTracer(provider, t.service_name)
