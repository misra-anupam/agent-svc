"""SSE framing and the domain-event -> pydantic contract mapper."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from collections.abc import AsyncIterator

from agent_svc.api.schemas.chat import EVENT_MODELS
from agent_svc.domain.models import AgentEvent
from agent_svc.domain.ports import RunBuffer

HEARTBEAT = b": ping\n\n"


def frame(event_id: int, event_type: str, json_data: str) -> bytes:
    return f"id: {event_id}\nevent: {event_type}\ndata: {json_data}\n\n".encode()


def to_wire(event: AgentEvent) -> tuple[str, str]:
    """Validates through the pydantic contract so the wire format is always schema-correct."""
    payload = dataclasses.asdict(event)
    model = EVENT_MODELS[event.type].model_validate(payload)
    return event.type, model.model_dump_json(exclude_none=True)


class RunSupervisor:
    """Runs agent streams as background tasks that write SSE frames to a RunBuffer.

    Clients tail the buffer, so a dropped connection can resume from ``Last-Event-ID`` and a
    repeated ``Idempotency-Key`` replays the original run instead of executing it twice. By
    default a disconnect cancels the run (stops paying for abandoned work); with
    ``detach_on_disconnect`` the run finishes in the background so it can be resumed.
    """

    def __init__(self, buffer: RunBuffer, *, detach_on_disconnect: bool) -> None:
        self._buffer = buffer
        self._detach = detach_on_disconnect
        self._tasks: dict[str, asyncio.Task[None]] = {}

    async def start(self, run_id: str, owner: str, events: AsyncIterator[AgentEvent]) -> None:
        await self._buffer.start(run_id, owner)
        task = asyncio.create_task(self._produce(run_id, events), name=f"run-{run_id}")
        self._tasks[run_id] = task
        task.add_done_callback(lambda _t: self._tasks.pop(run_id, None))

    async def _produce(self, run_id: str, events: AsyncIterator[AgentEvent]) -> None:
        seq = 0
        try:
            async for ev in events:
                seq += 1
                etype, data = to_wire(ev)
                await self._buffer.append(run_id, frame(seq, etype, data))
        finally:
            aclose = getattr(events, "aclose", None)
            if aclose:
                with contextlib.suppress(Exception):
                    await asyncio.shield(aclose())  # runs ChatService cleanup (lease release)
            await asyncio.shield(self._buffer.finish(run_id))

    async def owner(self, run_id: str) -> str | None:
        return await self._buffer.owner(run_id)

    async def tail(
        self, run_id: str, *, after: int = 0, heartbeat_seconds: float
    ) -> AsyncIterator[bytes]:
        idx = after
        try:
            while True:
                frames, done = await self._buffer.read(run_id, idx, heartbeat_seconds)
                if frames:
                    idx += len(frames)
                    for f in frames:
                        yield f
                elif done:
                    return
                else:
                    yield HEARTBEAT
        finally:
            if not self._detach:  # consumer gone (disconnect or finished): stop unfinished work
                task = self._tasks.get(run_id)
                if task and not task.done():
                    task.cancel()

    async def drain(self, grace_seconds: float) -> None:
        """Graceful shutdown: let in-flight runs finish, then cancel stragglers."""
        tasks = list(self._tasks.values())
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=grace_seconds)
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
