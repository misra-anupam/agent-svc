"""MLflow Tracing adapter (mlflow-tracing SDK): spans appear in the MLflow UI "Traces" tab."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import structlog

from agent_svc.config.settings import Settings
from agent_svc.domain.exceptions import ConfigurationError

log = structlog.get_logger()

_SPAN_TYPES = {
    "tool": "TOOL",
    "mcp": "TOOL",
    "rag": "RETRIEVER",
    "llm": "CHAT_MODEL",
    "agent": "AGENT",
    "guardrail": "UNKNOWN",
}


class MlflowSpan:
    def __init__(self, span: Any) -> None:
        self._span = span
        self.trace_id = str(getattr(span, "trace_id", ""))

    def set_attribute(self, key: str, value: Any) -> None:
        self._span.set_attribute(key, value)

    def record_error(self, exc: BaseException) -> None:
        self._span.add_event(_exception_event(exc))
        self._span.set_status("ERROR")

    def end(self) -> None:
        self._span.end()


def _exception_event(exc: BaseException) -> Any:
    from mlflow.entities import SpanEvent

    return SpanEvent.from_exception(exc if isinstance(exc, Exception) else RuntimeError(repr(exc)))


class MlflowTracer:
    def __init__(self, mlflow_module: Any) -> None:
        self._m = mlflow_module

    def start_span(
        self, name: str, *, parent: Any = None, attributes: dict[str, Any] | None = None
    ) -> MlflowSpan:
        attrs = dict(attributes or {})
        kind = str(attrs.get("step.kind", "agent"))
        span = self._m.start_span_no_context(
            name=name,
            span_type=_SPAN_TYPES.get(kind, "UNKNOWN"),
            parent_span=parent._span if isinstance(parent, MlflowSpan) else None,
            attributes=attrs,
        )
        return MlflowSpan(span)

    async def flush(self) -> None:
        flush = getattr(self._m, "flush_trace_async_logging", None)
        if flush:
            await asyncio.to_thread(flush)

    async def shutdown(self) -> None:
        await self.flush()


def build(settings: Settings) -> MlflowTracer:
    import mlflow

    t = settings.tracing
    if not t.mlflow_tracking_uri:
        raise ConfigurationError(
            "tracing.provider=mlflow requires AGENT_TRACING__MLFLOW_TRACKING_URI"
        )
    # A tracing outage must never block startup or requests: short timeouts, no long retry loops.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
    mlflow.set_tracking_uri(t.mlflow_tracking_uri)
    try:
        mlflow.set_experiment(t.mlflow_experiment)
    except Exception as exc:
        log.warning("mlflow_unreachable_at_startup", error=str(exc)[:200])
    return MlflowTracer(mlflow)
