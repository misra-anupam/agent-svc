from __future__ import annotations

from typing import Any

from agent_svc.config.settings import Settings


class NoopSpan:
    trace_id = ""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def record_error(self, exc: BaseException) -> None:
        pass

    def end(self) -> None:
        pass


class NoopTracer:
    def start_span(
        self, name: str, *, parent: Any = None, attributes: dict[str, Any] | None = None
    ) -> NoopSpan:
        return NoopSpan()

    async def flush(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass


def build(_settings: Settings) -> NoopTracer:
    return NoopTracer()
