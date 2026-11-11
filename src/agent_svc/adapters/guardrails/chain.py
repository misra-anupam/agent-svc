"""GuardrailChain: run rails in order. Redactions feed forward; the first block wins."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Literal

import structlog

from agent_svc.domain.models import Principal
from agent_svc.domain.ports import Guardrail, GuardrailResult, GuardrailStage
from agent_svc.observability.metrics import GUARDRAIL_CHECKS, GUARDRAIL_LATENCY

log = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class RailTrace:
    rail: str
    action: str
    reason: str
    ms: int


@dataclass(frozen=True, slots=True)
class ChainOutcome:
    result: GuardrailResult
    trail: tuple[RailTrace, ...] = field(default_factory=tuple)

    @property
    def blocked(self) -> bool:
        return self.result.action == "block"


class GuardrailChain:
    name = "chain"

    def __init__(
        self, rails: list[Guardrail], *, fail_mode: str = "closed", timeout_seconds: float = 10.0
    ) -> None:
        self._rails = rails
        self._timeout = timeout_seconds
        self._fail_closed = fail_mode == "closed"

    @property
    def rails(self) -> list[str]:
        return [r.name for r in self._rails]

    async def evaluate(
        self, text: str, *, stage: GuardrailStage, principal: Principal, context: str = ""
    ) -> ChainOutcome:
        current, redacted, trail = text, False, []
        for rail in self._rails:
            started = time.perf_counter()
            try:
                res = await asyncio.wait_for(
                    rail.check(current, stage=stage, principal=principal, context=context),
                    self._timeout,
                )
            except Exception as exc:  # a broken rail must not silently disable protection
                log.warning("guardrail_error", rail=rail.name, stage=stage.value, error=str(exc))
                action: Literal["allow", "block"] = "block" if self._fail_closed else "allow"
                res = GuardrailResult(
                    action, current, rail.name, f"rail_error:{type(exc).__name__}"
                )
            ms = int((time.perf_counter() - started) * 1000)
            GUARDRAIL_CHECKS.add(1, {"rail": rail.name, "stage": stage.value, "action": res.action})
            GUARDRAIL_LATENCY.record(ms / 1000, {"rail": rail.name, "stage": stage.value})
            trail.append(RailTrace(rail.name, res.action, res.reason, ms))
            if res.action == "block":
                return ChainOutcome(
                    GuardrailResult("block", rail=rail.name, reason=res.reason), tuple(trail)
                )
            if res.action == "redact":
                current, redacted = res.text, True
        final: Literal["allow", "redact"] = "redact" if redacted else "allow"
        return ChainOutcome(GuardrailResult(final, current, "chain"), tuple(trail))

    async def check(
        self, text: str, *, stage: GuardrailStage, principal: Principal, context: str = ""
    ) -> GuardrailResult:
        return (await self.evaluate(text, stage=stage, principal=principal, context=context)).result
