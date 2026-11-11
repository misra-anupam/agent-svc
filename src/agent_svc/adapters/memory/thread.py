from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Sequence

import redis.asyncio as aioredis

from agent_svc.domain.models import Message


def _enc(m: Message) -> str:
    return json.dumps({"r": m.role, "c": m.content, "n": m.name})


def _dec(raw: bytes | str) -> Message:
    d = json.loads(raw)
    return Message(d["r"], d["c"], d.get("n"))


class InMemoryThreadMemory:
    def __init__(self, max_messages: int = 200) -> None:
        self._d: dict[tuple[str, str], deque[Message]] = defaultdict(
            lambda: deque(maxlen=max_messages)
        )

    async def append(self, tenant: str, thread_id: str, messages: Sequence[Message]) -> None:
        self._d[(tenant, thread_id)].extend(messages)

    async def load(self, tenant: str, thread_id: str, limit: int | None = None) -> list[Message]:
        msgs = list(self._d.get((tenant, thread_id), ()))
        return msgs[-limit:] if limit else msgs

    async def delete(self, tenant: str, thread_id: str) -> None:
        self._d.pop((tenant, thread_id), None)


class RedisThreadMemory:
    def __init__(
        self, client: aioredis.Redis, *, ttl_seconds: int, max_messages: int, prefix: str = "thread"
    ) -> None:
        self._r, self._ttl, self._max, self._p = client, ttl_seconds, max_messages, prefix

    def _k(self, tenant: str, thread_id: str) -> str:
        return f"{self._p}:{tenant}:{thread_id}"

    async def append(self, tenant: str, thread_id: str, messages: Sequence[Message]) -> None:
        if not messages:
            return
        key = self._k(tenant, thread_id)
        pipe = self._r.pipeline()
        pipe.rpush(key, *[_enc(m) for m in messages])
        pipe.ltrim(key, -self._max, -1)
        pipe.expire(key, self._ttl)  # sliding TTL: active threads stay alive
        await pipe.execute()

    async def load(self, tenant: str, thread_id: str, limit: int | None = None) -> list[Message]:
        start = -limit if limit else 0
        return [_dec(r) for r in await self._r.lrange(self._k(tenant, thread_id), start, -1)]

    async def delete(self, tenant: str, thread_id: str) -> None:
        await self._r.delete(self._k(tenant, thread_id))
