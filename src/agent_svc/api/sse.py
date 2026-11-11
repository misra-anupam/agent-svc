"""SSE framing and the domain-event -> pydantic contract mapper."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from collections.abc import AsyncIterator

from agent_svc.api.schemas.chat import EVENT_MODELS
from agent_svc.domain.models import AgentEvent

HEARTBEAT = b": ping\n\n"


def frame(event_id: int, event_type: str, json_data: str) -> bytes:
    return f"id: {event_id}\nevent: {event_type}\ndata: {json_data}\n\n".encode()


def to_wire(event: AgentEvent) -> tuple[str, str]:
    """Validates through the pydantic contract so the wire format is always schema-correct."""
    payload = dataclasses.asdict(event)
    model = EVENT_MODELS[event.type].model_validate(payload)
    return event.type, model.model_dump_json(exclude_none=True)


async def sse_stream(
    events: AsyncIterator[AgentEvent], *, heartbeat_seconds: float
) -> AsyncIterator[bytes]:
    """Pumps events through a queue so heartbeats flow during long silent tool calls.

    If the client disconnects, the generator is closed and the producer task is cancelled,
    which propagates cancellation into the agent run.
    """
    queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue(maxsize=64)

    async def produce() -> None:
        try:
            async for ev in events:
                await queue.put(ev)
        finally:
            await queue.put(None)

    task = asyncio.create_task(produce())
    seq = 0
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=heartbeat_seconds)
            except TimeoutError:
                yield HEARTBEAT
                continue
            if item is None:
                return
            seq += 1
            etype, data = to_wire(item)
            yield frame(seq, etype, data)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        aclose = getattr(events, "aclose", None)
        if aclose:
            with contextlib.suppress(Exception):
                await aclose()
