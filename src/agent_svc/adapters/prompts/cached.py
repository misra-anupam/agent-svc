"""TTL cache decorator for any PromptStore. Pinned versions are immutable, so they cache forever;
aliases and "latest" can move, so they expire after ``ttl``."""

from __future__ import annotations

import time

from agent_svc.domain.ports import PromptStore, PromptVersion


class CachedPromptStore:
    def __init__(self, inner: PromptStore, ttl_seconds: float, *, clock=time.monotonic) -> None:  # type: ignore[no-untyped-def]
        self._inner, self._ttl, self._clock = inner, ttl_seconds, clock
        self._cache: dict[tuple[str, str | None, str | None], tuple[float, PromptVersion]] = {}

    async def get(
        self, name: str, *, version: str | None = None, alias: str | None = None
    ) -> PromptVersion:
        key = (name, version, alias)
        hit = self._cache.get(key)
        now = self._clock()
        if hit and (version is not None or now - hit[0] < self._ttl):
            return hit[1]
        pv = await self._inner.get(name, version=version, alias=alias)
        self._cache[key] = (now, pv)
        return pv

    async def register(self, name: str, template: str) -> PromptVersion:
        pv = await self._inner.register(name, template)
        self._cache = {k: v for k, v in self._cache.items() if k[0] != name}
        return pv

    async def set_alias(self, name: str, alias: str, version: str) -> None:
        await self._inner.set_alias(name, alias, version)
        self._cache = {k: v for k, v in self._cache.items() if k[0] != name}
