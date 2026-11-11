"""Admission control: rate limit (per tier) -> token budgets (user + IP) -> concurrency lease."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass

import structlog

from agent_svc.config.settings import RateLimitSettings, TierLimit
from agent_svc.domain.exceptions import BudgetExceededError, RateLimitedError, UnavailableError
from agent_svc.domain.models import Principal, Usage
from agent_svc.domain.ports import ConcurrencyGate, RateLimiter, TokenBudget
from agent_svc.observability.metrics import RATE_LIMITED

log = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class Admission:
    gate_key: str | None = None
    lease: str | None = None


class AdmissionService:
    def __init__(
        self,
        settings: RateLimitSettings,
        *,
        limiter: RateLimiter | None,
        budget: TokenBudget | None,
        gate: ConcurrencyGate | None,
    ) -> None:
        self._s, self._limiter, self._budget, self._gate = settings, limiter, budget, gate

    def tier_for(self, principal: Principal) -> tuple[str, TierLimit]:
        name = principal.tier if principal.tier in self._s.tiers else self._s.default_tier
        return name, self._s.tiers[name]  # unknown tier claims fall back to the default (lowest)

    async def admit(self, principal: Principal) -> Admission:
        tier_name, tier = self.tier_for(principal)
        try:
            await self._rate_limit(principal, tier_name, tier)
            await self._check_budgets(principal, tier_name, tier)
            return await self._acquire_slot(principal, tier_name, tier)
        except (RateLimitedError, UnavailableError):
            raise
        except Exception as exc:  # limiter backend failure
            if self._s.fail_open:
                log.warning("admission_backend_failed_open", error=str(exc))
                return Admission()
            raise UnavailableError("admission control unavailable") from exc

    async def _rate_limit(self, p: Principal, tier_name: str, tier: TierLimit) -> None:
        if self._limiter is None:
            return
        for attempt in (0, 1):
            d = await self._limiter.acquire(
                p.rate_key, rate_per_minute=tier.requests_per_minute, burst=tier.burst
            )
            if d.allowed:
                return
            delay_ms = d.retry_after_seconds * 1000
            if (
                attempt == 0
                and self._s.mode == "throttle"
                and delay_ms <= self._s.max_throttle_delay_ms
            ):
                RATE_LIMITED.add(1, {"reason": "throttled", "tier": tier_name})
                await asyncio.sleep(d.retry_after_seconds)  # throttle: delay instead of reject
                continue
            RATE_LIMITED.add(1, {"reason": "rate", "tier": tier_name})
            raise RateLimitedError(
                "rate limit exceeded", retry_after=max(1, math.ceil(d.retry_after_seconds))
            )

    async def _check_budgets(self, p: Principal, tier_name: str, tier: TierLimit) -> None:
        if self._budget is None:
            return
        checks = [(f"tok:{p.rate_key}", tier.daily_token_budget)]
        if p.ip:
            checks.append((f"tok:ip:{p.ip}", self._s.ip_daily_token_budget))
        for key, limit in checks:
            d = await self._budget.check(key, limit=limit)
            if not d.allowed:
                RATE_LIMITED.add(1, {"reason": "budget", "tier": tier_name})
                raise BudgetExceededError(
                    "daily token budget exhausted",
                    retry_after=max(1, math.ceil(d.retry_after_seconds)),
                )

    async def _acquire_slot(self, p: Principal, tier_name: str, tier: TierLimit) -> Admission:
        if self._gate is None:
            return Admission()
        lease = await self._gate.acquire(
            p.rate_key, limit=tier.max_concurrent, lease_seconds=self._s.concurrency_lease_seconds
        )
        if lease is None:
            RATE_LIMITED.add(1, {"reason": "concurrency", "tier": tier_name})
            raise RateLimitedError("too many concurrent runs", retry_after=1)
        return Admission(gate_key=p.rate_key, lease=lease)

    async def release(self, adm: Admission) -> None:
        if self._gate and adm.lease and adm.gate_key:
            try:
                await self._gate.release(adm.gate_key, adm.lease)
            except Exception:
                log.warning("lease_release_failed", exc_info=True)  # lease TTL cleans up

    async def record_usage(self, p: Principal, usage: Usage) -> None:
        if self._budget is None or usage.total_tokens <= 0:
            return
        _, tier = self.tier_for(p)
        try:
            await self._budget.consume(
                f"tok:{p.rate_key}", usage.total_tokens, limit=tier.daily_token_budget
            )
            if p.ip:
                await self._budget.consume(
                    f"tok:ip:{p.ip}", usage.total_tokens, limit=self._s.ip_daily_token_budget
                )
        except Exception:
            log.warning("usage_record_failed", exc_info=True)
