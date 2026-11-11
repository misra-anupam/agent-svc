"""Composition root: builds adapters from Settings, owns startup/shutdown order."""

from __future__ import annotations

import asyncio
from contextlib import suppress

import structlog

from agent_svc.adapters.infra.postgres import PostgresResource
from agent_svc.adapters.infra.redis import RedisResource
from agent_svc.adapters.registries import build_runner
from agent_svc.config.settings import Settings
from agent_svc.domain.exceptions import UnavailableError
from agent_svc.domain.ports import AgentRunner, HealthCheck, ReadinessReport, Resource

log = structlog.get_logger()


class Container:
    def __init__(
        self, settings: Settings, *, extra_resources: list[Resource] | None = None
    ) -> None:
        self.settings = settings
        self.resources: list[Resource] = []
        self.checks: list[HealthCheck] = []
        self.runner: AgentRunner | None = None
        self._extra = extra_resources or []
        self._started: list[Resource] = []

    def _register(self, res: Resource) -> None:
        self.resources.append(res)
        if isinstance(res, HealthCheck):
            self.checks.append(res)

    async def start(self) -> None:
        s = self.settings
        if s.redis.enabled:
            self._register(RedisResource(s.redis))
        if s.postgres.enabled:
            self._register(PostgresResource(s.postgres))
        for res in self._extra:
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

        self.runner = build_runner(s)
        log.info("container_ready", orchestrator=s.orchestrator.provider)

    async def stop(self) -> None:
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
