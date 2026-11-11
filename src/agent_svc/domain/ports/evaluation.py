from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class GoldenCase:
    id: str
    input: str
    expected: str | None = None
    expected_tools: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvalReport:
    evaluator: str
    agent_version: str
    metrics: dict[str, float]
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


class Evaluator(Protocol):
    async def evaluate(
        self, cases: Sequence[GoldenCase], *, agent_version: str, thresholds: dict[str, float]
    ) -> EvalReport: ...
