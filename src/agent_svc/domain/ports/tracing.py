from __future__ import annotations

from typing import Any, Protocol


class Span(Protocol):
    trace_id: str

    def set_attribute(self, key: str, value: Any) -> None: ...
    def record_error(self, exc: BaseException) -> None: ...
    def end(self) -> None: ...


class Tracer(Protocol):
    """Backend-neutral tracing. Implementations: otel, mlflow, langfuse, noop."""

    def start_span(
        self, name: str, *, parent: Span | None = None, attributes: dict[str, Any] | None = None
    ) -> Span: ...

    async def flush(self) -> None: ...
    async def shutdown(self) -> None: ...
