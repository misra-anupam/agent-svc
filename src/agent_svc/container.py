"""Composition root: builds adapters from Settings, owns startup/shutdown order."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

import structlog

from agent_svc.adapters import registries as reg
from agent_svc.adapters.cache.cached_runner import CachedRunner
from agent_svc.adapters.cache.semantic import SemanticIndex
from agent_svc.adapters.cache.stores import InMemoryCache, RedisCache
from agent_svc.adapters.guardrails.guarded_runner import GuardedRunner
from agent_svc.adapters.infra import Infra
from agent_svc.adapters.infra.postgres import PostgresResource
from agent_svc.adapters.infra.redis import RedisResource
from agent_svc.adapters.infra.schema import ensure_schema
from agent_svc.adapters.tools.composite import CompositeToolProvider, GuardedToolProvider
from agent_svc.adapters.tracing.traced_runner import TracedRunner
from agent_svc.application.admission import AdmissionService
from agent_svc.application.memory_facade import MemoryFacade
from agent_svc.config.settings import Settings
from agent_svc.domain.exceptions import UnavailableError
from agent_svc.domain.ports import (
    AgentRunner,
    Authenticator,
    Cache,
    HealthCheck,
    LLMClient,
    ReadinessReport,
    Resource,
    RunBuffer,
    ToolProvider,
    Tracer,
)

log = structlog.get_logger()


def parse_prompt_ref(ref: str) -> tuple[str, str | None, str | None]:
    """``name`` | ``name:version`` | ``name@alias`` -> (name, version, alias)."""
    if "@" in ref:
        name, _, alias = ref.partition("@")
        return name, None, alias
    if ":" in ref:
        name, _, version = ref.partition(":")
        return name, version, None
    return ref, None, None


class Container:
    def __init__(
        self, settings: Settings, *, extra_resources: list[Resource] | None = None
    ) -> None:
        self.settings = settings
        self.resources: list[Resource] = []
        self.checks: list[HealthCheck] = []
        self.llm: LLMClient | None = None
        self.runner: AgentRunner | None = None
        self.authenticator: Authenticator | None = None
        self.admission: AdmissionService | None = None
        self.memory: MemoryFacade = MemoryFacade(settings.memory)
        self.run_buffer: RunBuffer | None = None
        self.tracer: Tracer | None = None
        self.tools: ToolProvider | None = None
        self._tool_sources: list[ToolProvider] = []
        self._guard_policy: Any = None
        self.prompt_ref: str | None = None  # e.g. "system:3" when the prompt came from a store
        self.kv: Cache | None = None  # small key/value store (idempotency keys)
        self._extra = extra_resources or []
        self._started: list[Resource] = []
        self._redis: RedisResource | None = None
        self._pg: PostgresResource | None = None

    def _register(self, res: Resource) -> None:
        self.resources.append(res)
        if isinstance(res, HealthCheck):
            self.checks.append(res)

    async def start(self) -> None:
        s = self.settings
        if s.redis.enabled:
            self._redis = RedisResource(s.redis)
            self._register(self._redis)
        if s.postgres.enabled:
            self._pg = PostgresResource(s.postgres)
            self._register(self._pg)
        self._tool_sources = reg.build_tool_sources(s)  # constructed now, connected below
        for src in self._tool_sources:
            if isinstance(src, Resource):
                self._register(src)
        for res in self._extra:
            if isinstance(res, RedisResource) and self._redis is None:
                self._redis = res
            self._register(res)

        for res in self.resources:  # ordered startup
            try:
                await res.start()
            except Exception as exc:
                if getattr(res, "required", False):
                    await self.stop()
                    raise UnavailableError(f"required resource {res.name!r} failed: {exc}") from exc
                log.warning("resource_start_failed", resource=res.name, error=str(exc))
                continue
            self._started.append(res)
            log.info("resource_started", resource=res.name)

        try:
            await self._build()
        except BaseException:
            await self.stop()
            raise
        log.info("container_ready", orchestrator=s.orchestrator.provider)

    async def _load_prompt(self) -> None:
        """Replace the system prompt with the store's version (``prompts.ref``), if configured."""
        ref = self.settings.prompts.ref
        if not ref:
            return
        name, version, alias = parse_prompt_ref(ref)
        pv = await reg.build_prompt_store(self.settings).get(name, version=version, alias=alias)
        self.settings = self.settings.model_copy(deep=True)  # never mutate the caller's settings
        self.settings.orchestrator.system_prompt = pv.template
        self.prompt_ref = pv.ref
        log.info("prompt_loaded", ref=pv.ref, requested=ref)

    async def _build(self) -> None:
        await self._load_prompt()
        s = self.settings
        redis = self._redis.client if self._redis and self._redis in self._started else None
        pool = self._pg.pool if self._pg and self._pg in self._started else None
        infra = Infra(redis=redis, pg_pool=pool)

        self.authenticator = reg.authenticators.create("jwt" if s.auth.enabled else "noauth", s)

        base_llm = reg.build_llm(s)
        self.llm = base_llm
        self.tools = GuardedToolProvider(
            CompositeToolProvider(self._tool_sources),
            allowlist=s.tools.allowlist,
            timeout_seconds=s.tools.timeout_seconds,
            max_result_chars=s.tools.max_result_chars,
        )
        runner = reg.build_runner(s, base_llm, self.tools)
        if s.guardrails.enabled:  # inside the cache: cached answers were already output-checked
            self._guard_policy = reg.build_guardrail_policy(s)
            runner = GuardedRunner(runner, self._guard_policy, s.guardrails)
        if s.cache.provider != "none":
            semantic = (
                SemanticIndex(
                    reg.embedders.create(s.memory.embedding_provider, s),
                    threshold=s.cache.semantic_threshold,
                    max_entries=s.cache.semantic_max_entries,
                )
                if s.cache.semantic
                else None
            )
            runner = CachedRunner(
                runner, reg.caches.create(s.cache.provider, s, infra), s.cache, semantic
            )
        self.tracer = reg.tracers.create(s.tracing.provider, s)
        self.runner = TracedRunner(runner, self.tracer, s.tracing)  # outermost: sees cache hits too

        backend = None
        if s.ratelimit.provider != "none":
            backend = reg.limit_backends.create(s.ratelimit.provider, s, infra)
        self.admission = AdmissionService(
            s.ratelimit,
            limiter=backend.limiter if backend else None,
            budget=backend.budget if backend else None,
            gate=backend.gate if backend else None,
        )

        m = s.memory
        if pool is not None and (
            m.user_provider == "postgres" or m.long_term_provider == "pgvector"
        ):
            await ensure_schema(
                pool, embedding_dim=m.embedding_dim, vector=m.long_term_provider == "pgvector"
            )
        embedder = (
            reg.embedders.create(m.embedding_provider, s)
            if m.long_term_provider != "none"
            else None
        )
        self.memory = MemoryFacade(
            m,
            thread=reg.thread_memories.create(m.thread_provider, s, infra)
            if m.thread_provider != "none"
            else None,
            user=reg.user_memories.create(m.user_provider, s, infra)
            if m.user_provider != "none"
            else None,
            long_term=(
                reg.long_term_memories.create(m.long_term_provider, s, infra, embedder)
                if embedder is not None
                else None
            ),
        )

        self.run_buffer = reg.run_buffers.create(s.app.run_buffer, s, infra)
        self.kv = RedisCache(redis, "kv") if redis is not None else InMemoryCache()

    async def stop(self) -> None:
        if self._guard_policy is not None:
            with suppress(Exception):
                await self._guard_policy.aclose()
            self._guard_policy = None
        if self.tracer is not None:
            with suppress(Exception):
                await asyncio.wait_for(self.tracer.shutdown(), timeout=10)  # flush spans
            self.tracer = None
        for res in reversed(self._started):  # reverse-order teardown
            with suppress(Exception):
                await asyncio.wait_for(res.stop(), timeout=self.settings.app.shutdown_grace_seconds)
            log.info("resource_stopped", resource=res.name)
        self._started.clear()

    async def readiness(self) -> ReadinessReport:
        async def probe(check: HealthCheck) -> tuple[str, bool]:
            try:
                return check.name, await asyncio.wait_for(check.ping(), timeout=2.0)
            except Exception:
                return check.name, False

        results = dict(await asyncio.gather(*(probe(c) for c in self.checks)))
        ready = all(results[c.name] or not c.required for c in self.checks)
        return ReadinessReport(ready=ready, checks=results)
