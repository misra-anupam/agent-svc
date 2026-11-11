from __future__ import annotations

import json
from typing import Any

from agent_svc.domain.models import AgentRequest, Message, Principal


def make_request(text: str = "hello world", run_id: str = "run1") -> AgentRequest:
    return AgentRequest(
        run_id=run_id,
        thread_id="t1",
        principal=Principal(subject="u1"),
        messages=(Message("user", text),),
        agent_name="default",
        agent_version="1.0.0",
    )


def parse_sse(body: str) -> list[tuple[str, dict[str, object]]]:
    out: list[tuple[str, dict[str, object]]] = []
    for block in body.split("\n\n"):
        if not block.strip() or block.startswith(":"):
            continue
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        out.append((fields["event"], json.loads(fields["data"])))
    return out


class RecordingTracer:
    """Recording tracer used to assert the span tree independent of any backend."""

    def __init__(self) -> None:
        self.spans: list[RecordingTracer.S] = []

    class S:
        def __init__(self, name: str, parent: Any, attrs: dict[str, Any]) -> None:
            self.name, self.parent, self.attrs = name, parent, dict(attrs)
            self.ended, self.error, self.trace_id = False, None, "abc123"

        def set_attribute(self, k: str, v: Any) -> None:
            self.attrs[k] = v

        def record_error(self, exc: BaseException) -> None:
            self.error = str(exc)

        def end(self) -> None:
            self.ended = True

    def start_span(
        self, name: str, *, parent: Any = None, attributes: Any = None
    ) -> RecordingTracer.S:
        s = RecordingTracer.S(name, parent, attributes or {})
        self.spans.append(s)
        return s

    async def flush(self) -> None: ...
    async def shutdown(self) -> None: ...
