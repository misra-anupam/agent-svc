"""CachedRunner: answer cache decorator for any AgentRunner (exact-match, with single-flight)."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator

import structlog

from agent_svc.adapters.cache.semantic import SemanticIndex
from agent_svc.config.settings import CacheSettings
from agent_svc.domain.models import (
    AgentEvent,
    AgentRequest,
    MessageCompleted,
    MessageDelta,
    RunCompleted,
    RunStarted,
    Usage,
)
from agent_svc.domain.ports import AgentRunner, Cache
from agent_svc.observability.metrics import CACHE_REQUESTS

log = structlog.get_logger()


def cache_key(req: AgentRequest, scope: str) -> str:
    owner = {"user": req.principal.subject, "tenant": req.principal.tenant, "global": ""}[scope]
    material = json.dumps(
        [
            req.agent_name,
            req.agent_version,  # a version bump invalidates every cached answer
            req.principal.tenant,
            owner,
            [(m.role, " ".join(m.content.split())) for m in req.messages],  # whitespace-normalised
        ],
        ensure_ascii=False,
    )
    return hashlib.sha256(material.encode()).hexdigest()


def semantic_partition(req: AgentRequest, scope: str) -> str:
    """Agent version + scope + earlier conversation: the only place a similar hit may come from."""
    owner = {"user": req.principal.subject, "tenant": req.principal.tenant, "global": ""}[scope]
    history = [(m.role, m.content) for m in req.messages[:-1]]
    material = json.dumps(
        [req.agent_name, req.agent_version, req.principal.tenant, owner, history],
        ensure_ascii=False,
    )
    return hashlib.sha256(material.encode()).hexdigest()


class CachedRunner:
    def __init__(
        self,
        inner: AgentRunner,
        cache: Cache,
        settings: CacheSettings,
        semantic: SemanticIndex | None = None,
    ) -> None:
        self._inner, self._cache, self._s = inner, cache, settings
        self._semantic = semantic
        self._inflight: dict[str, asyncio.Future[bytes | None]] = {}

    def _bypass(self, req: AgentRequest) -> bool:
        return bool(req.metadata.get("no_cache"))

    async def stream(self, request: AgentRequest) -> AsyncIterator[AgentEvent]:
        if self._bypass(request):
            CACHE_REQUESTS.add(1, {"result": "bypass"})
            async for ev in self._inner.stream(request):
                yield ev
            return

        key = cache_key(request, self._s.scope)
        hit = await self._lookup(key)
        if hit is not None:
            CACHE_REQUESTS.add(1, {"result": "hit"})
            for ev in self._replay(request, hit):
                yield ev
            return

        partition = semantic_partition(request, self._s.scope) if self._semantic else ""
        if self._semantic is not None:
            similar = await self._semantic_lookup(partition, request.last_user_text)
            if similar is not None:
                CACHE_REQUESTS.add(1, {"result": "semantic_hit"})
                for ev in self._replay(request, similar):
                    yield ev
                return

        # single-flight: identical concurrent requests wait for the leader instead of all paying
        existing = self._inflight.get(key)
        fut: asyncio.Future[bytes | None] | None = None
        if existing is not None:
            shared = await asyncio.shield(existing)
            if shared is not None:
                CACHE_REQUESTS.add(1, {"result": "coalesced"})
                for ev in self._replay(request, shared):
                    yield ev
                return
            # leader failed or produced nothing cacheable: run independently below
        else:
            fut = asyncio.get_running_loop().create_future()
            self._inflight[key] = fut

        CACHE_REQUESTS.add(1, {"result": "miss"})
        text, usage, ok = "", Usage(), False
        try:
            async for ev in self._inner.stream(request):
                if isinstance(ev, MessageCompleted):
                    text = ev.text
                elif isinstance(ev, RunCompleted):
                    usage, ok = ev.usage, True
                yield ev
        finally:
            payload: bytes | None = None
            if ok and text:
                payload = json.dumps(
                    {"text": text, "in": usage.input_tokens, "out": usage.output_tokens}
                ).encode()
                await self._cache.set(key, payload, self._s.ttl_seconds)
                if self._semantic is not None:
                    await self._semantic_add(partition, request.last_user_text, key)
            if fut is not None:
                self._inflight.pop(key, None)
                if not fut.done():
                    fut.set_result(payload)

    async def _semantic_lookup(self, partition: str, text: str) -> bytes | None:
        assert self._semantic is not None
        try:
            key = await self._semantic.find(partition, text)
            return await self._cache.get(key) if key else None
        except Exception:  # an embedder/cache outage must never break answering
            log.warning("semantic_lookup_failed", exc_info=True)
            return None

    async def _semantic_add(self, partition: str, text: str, key: str) -> None:
        assert self._semantic is not None
        try:
            await self._semantic.add(partition, text, key)
        except Exception:
            log.warning("semantic_index_failed", exc_info=True)

    async def _lookup(self, key: str) -> bytes | None:
        try:
            return await self._cache.get(key)
        except Exception:  # cache outage must never break answering
            log.warning("cache_lookup_failed", exc_info=True)
            return None

    @staticmethod
    def _replay(req: AgentRequest, raw: bytes) -> list[AgentEvent]:
        d = json.loads(raw)
        rid = req.run_id
        return [
            RunStarted(
                run_id=rid,
                thread_id=req.thread_id,
                agent_name=req.agent_name,
                agent_version=req.agent_version,
            ),
            MessageDelta(run_id=rid, text=d["text"]),
            MessageCompleted(run_id=rid, text=d["text"]),
            RunCompleted(run_id=rid, usage=Usage(), cached=True),
        ]
