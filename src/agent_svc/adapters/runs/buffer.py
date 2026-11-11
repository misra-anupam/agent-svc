"""RunBuffer implementations: in-memory (asyncio) and Redis (list + polling tail)."""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field

import redis.asyncio as aioredis


def _b(v: bytes | str) -> bytes:
    return v if isinstance(v, bytes) else v.encode()


@dataclass
class _Run:
    owner: str
    frames: list[bytes] = field(default_factory=list)
    done: bool = False
    cond: asyncio.Condition = field(default_factory=asyncio.Condition)
    created: float = field(default_factory=time.monotonic)


class InMemoryRunBuffer:
    def __init__(self, ttl_seconds: int = 3600, max_runs: int = 1000) -> None:
        self._runs: dict[str, _Run] = {}
        self._ttl, self._max = ttl_seconds, max_runs

    def _gc(self) -> None:
        now = time.monotonic()
        for rid in [r for r, v in self._runs.items() if v.done and now - v.created > self._ttl]:
            del self._runs[rid]
        while len(self._runs) > self._max:
            self._runs.pop(next(iter(self._runs)))

    async def start(self, run_id: str, owner: str) -> None:
        self._gc()
        self._runs[run_id] = _Run(owner)

    async def append(self, run_id: str, frame: bytes) -> None:
        run = self._runs[run_id]
        async with run.cond:
            run.frames.append(frame)
            run.cond.notify_all()

    async def finish(self, run_id: str) -> None:
        run = self._runs.get(run_id)
        if run:
            async with run.cond:
                run.done = True
                run.cond.notify_all()

    async def owner(self, run_id: str) -> str | None:
        run = self._runs.get(run_id)
        return run.owner if run else None

    async def read(self, run_id: str, after: int, wait_seconds: float) -> tuple[list[bytes], bool]:
        run = self._runs.get(run_id)
        if run is None:
            return [], True
        async with run.cond:
            if len(run.frames) <= after and not run.done:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(run.cond.wait(), wait_seconds)
            frames = run.frames[after:]
            return frames, run.done and not frames


class RedisRunBuffer:
    POLL = 0.05

    def __init__(
        self, client: aioredis.Redis, ttl_seconds: int = 3600, prefix: str = "run"
    ) -> None:
        self._r, self._ttl, self._p = client, ttl_seconds, prefix

    def _k(self, run_id: str, part: str) -> str:
        return f"{self._p}:{run_id}:{part}"

    async def start(self, run_id: str, owner: str) -> None:
        await self._r.set(self._k(run_id, "owner"), owner, ex=self._ttl)

    async def append(self, run_id: str, frame: bytes) -> None:
        key = self._k(run_id, "frames")
        pipe = self._r.pipeline()
        pipe.rpush(key, frame)
        pipe.expire(key, self._ttl)
        await pipe.execute()

    async def finish(self, run_id: str) -> None:
        await self._r.set(self._k(run_id, "done"), 1, ex=self._ttl)

    async def owner(self, run_id: str) -> str | None:
        v = await self._r.get(self._k(run_id, "owner"))
        return _b(v).decode() if v is not None else None

    async def read(self, run_id: str, after: int, wait_seconds: float) -> tuple[list[bytes], bool]:
        deadline = time.monotonic() + wait_seconds
        while True:
            # read `done` BEFORE frames so frames appended just before finish are never missed
            done = bool(await self._r.exists(self._k(run_id, "done")))
            frames = [_b(f) for f in await self._r.lrange(self._k(run_id, "frames"), after, -1)]
            if frames:
                return frames, False
            if done:
                return [], True
            if time.monotonic() >= deadline:
                return [], False
            await asyncio.sleep(self.POLL)
