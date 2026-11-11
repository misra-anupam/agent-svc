"""Langfuse tracer (SDK v3+, OpenTelemetry based)."""

from __future__ import annotations

import asyncio
from typing import Any

from agent_svc.config.settings import Settings
from agent_svc.domain.exceptions import ConfigurationError

_AS_TYPE = {
    "tool": "tool",
    "mcp": "tool",
    "rag": "retriever",
    "llm": "generation",
    "agent": "agent",
    "guardrail": "guardrail",
}


class LangfuseSpan:
    def __init__(self, obs: Any, kind: str) -> None:
        self._obs, self._kind = obs, kind
        self._meta: dict[str, Any] = {}
        self._error: str | None = None
        self.trace_id = str(getattr(obs, "trace_id", ""))

    def set_attribute(self, key: str, value: Any) -> None:
        self._meta[key] = value

    def record_error(self, exc: BaseException) -> None:
        self._error = f"{type(exc).__name__}: {exc}"

    def end(self) -> None:
        update: dict[str, Any] = {"metadata": self._meta}
        if self._error:
            update.update(level="ERROR", status_message=self._error)
        if self._kind == "llm":
            update["model"] = self._meta.get("gen_ai.request.model")
            update["usage_details"] = {
                "input": int(self._meta.get("gen_ai.usage.input_tokens", 0)),
                "output": int(self._meta.get("gen_ai.usage.output_tokens", 0)),
            }
            if "cost_usd" in self._meta:
                update["cost_details"] = {"total": float(self._meta["cost_usd"])}
        self._obs.update(**update)
        self._obs.end()


class LangfuseTracer:
    def __init__(self, client: Any) -> None:
        self._c = client

    def start_span(
        self, name: str, *, parent: Any = None, attributes: dict[str, Any] | None = None
    ) -> LangfuseSpan:
        attrs = dict(attributes or {})
        kind = str(attrs.get("step.kind", "agent"))
        as_type = _AS_TYPE.get(kind, "span")
        if isinstance(parent, LangfuseSpan):
            obs = parent._obs.start_observation(name=name, as_type=as_type)
        else:
            obs = self._c.start_observation(name=name, as_type=as_type)
        span = LangfuseSpan(obs, kind)
        for k, v in attrs.items():
            span.set_attribute(k, v)
        return span

    async def flush(self) -> None:
        await asyncio.to_thread(self._c.flush)

    async def shutdown(self) -> None:
        await asyncio.to_thread(self._c.shutdown)


def build(settings: Settings) -> LangfuseTracer:
    from langfuse import Langfuse

    t = settings.tracing
    if not (t.langfuse_public_key and t.langfuse_secret_key):
        raise ConfigurationError(
            "tracing.provider=langfuse requires AGENT_TRACING__LANGFUSE_PUBLIC_KEY and _SECRET_KEY"
        )
    return LangfuseTracer(
        Langfuse(
            public_key=t.langfuse_public_key.get_secret_value(),
            secret_key=t.langfuse_secret_key.get_secret_value(),
            host=t.langfuse_host,
            environment=settings.app.environment,
            release=settings.app.agent_version,
        )
    )
