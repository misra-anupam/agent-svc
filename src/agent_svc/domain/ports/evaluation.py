from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class GoldenCase:
    id: str
    input: str
    expected: str | None = None  # substring the answer must contain (case-insensitive)
    expected_tools: tuple[str, ...] = ()  # exact ordered tool-call trajectory, if any
    expect_error: str | None = None  # e.g. "policy_violation" for a prompt that must be blocked
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvalRecord:
    """What the agent actually did for one golden case."""

    case: GoldenCase
    output: str = ""
    tools_called: tuple[str, ...] = ()
    error_code: str | None = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True, slots=True)
class Threshold:
    min: float | None = None
    max: float | None = None


@dataclass(frozen=True, slots=True)
class EvalReport:
    evaluator: str
    agent_version: str
    metrics: dict[str, float]
    passed: bool
    failures: tuple[str, ...] = ()
    per_case: dict[str, dict[str, float]] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


class Evaluator(Protocol):
    name: str

    async def evaluate(
        self,
        records: Sequence[EvalRecord],
        *,
        agent_version: str,
        thresholds: dict[str, Threshold],
    ) -> EvalReport: ...
