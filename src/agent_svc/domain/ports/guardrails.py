from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol

from agent_svc.domain.models import Principal


class GuardrailStage(StrEnum):
    INPUT = "input"
    OUTPUT = "output"


@dataclass(frozen=True, slots=True)
class GuardrailResult:
    action: Literal["allow", "redact", "block"] = "allow"
    text: str = ""
    rail: str = ""
    reason: str = ""


class Guardrail(Protocol):
    name: str

    async def check(
        self, text: str, *, stage: GuardrailStage, principal: Principal
    ) -> GuardrailResult: ...
