from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
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
        self,
        name: str,
        *,
        parent: Any = None,
        attributes: Any = None,
        remote_parent: str | None = None,
    ) -> RecordingTracer.S:
        s = RecordingTracer.S(name, parent, attributes or {})
        s.remote_parent = remote_parent
        self.spans.append(s)
        return s

    async def flush(self) -> None: ...
    async def shutdown(self) -> None: ...


def metric_total(reader: Any, name: str, **attrs: str) -> float:
    """Sum of a counter's data points whose attributes include ``attrs`` (cumulative)."""
    data = reader.get_metrics_data()
    total = 0.0
    for rm in data.resource_metrics if data else []:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                if m.name != name:
                    continue
                for p in m.data.data_points:
                    if all(p.attributes.get(k) == v for k, v in attrs.items()):
                        total += getattr(p, "value", getattr(p, "sum", 0.0))
    return total


class HashEmbedder:
    """Deterministic hashed bag-of-words embedder: similar texts are close (tests only)."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in text.lower().split():
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]
