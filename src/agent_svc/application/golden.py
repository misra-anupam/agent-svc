"""Golden dataset: load cases + thresholds, replay them through an AgentRunner, record results.

Depends only on ports. Which evaluator scores the records is chosen by the caller.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from agent_svc.domain.exceptions import BadRequestError, DomainError
from agent_svc.domain.models import (
    AgentRequest,
    Message,
    MessageCompleted,
    Principal,
    RunError,
    StepKind,
    StepStarted,
    Usage,
    UsageReported,
)
from agent_svc.domain.ports import AgentRunner, EvalRecord, GoldenCase, Threshold


def load_cases(
    path: str | Path, *, tags: Iterable[str] = (), exclude_tags: Iterable[str] = ()
) -> list[GoldenCase]:
    """Read ``*.jsonl`` (a file, or every file in a directory). Ids must be unique."""
    p = Path(path)
    files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    if not files:
        raise BadRequestError(f"no golden dataset files under {p}")
    want, skip = set(tags), set(exclude_tags)
    cases: list[GoldenCase] = []
    seen: set[str] = set()
    for f in files:
        for n, line in enumerate(f.read_text().splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                d = json.loads(line)
                case = GoldenCase(
                    id=d["id"],
                    input=d["input"],
                    expected=d.get("expected"),
                    expected_tools=tuple(d.get("expected_tools", ())),
                    expect_error=d.get("expect_error"),
                    tags=tuple(d.get("tags", ())),
                    metadata=d.get("metadata", {}),
                )
            except (ValueError, KeyError, TypeError) as exc:
                raise BadRequestError(f"{f}:{n}: invalid golden case ({exc!r})") from exc
            if case.id in seen:
                raise BadRequestError(f"{f}:{n}: duplicate case id {case.id!r}")
            seen.add(case.id)
            if (want and not want & set(case.tags)) or skip & set(case.tags):
                continue
            cases.append(case)
    return cases


def load_thresholds(path: str | Path) -> dict[str, Threshold]:
    """YAML ``metric: {min: x, max: y}``."""
    raw: Any = yaml.safe_load(Path(path).read_text()) or {}
    out: dict[str, Threshold] = {}
    for name, spec in raw.items():
        if not isinstance(spec, dict) or not set(spec) <= {"min", "max"} or not spec:
            raise BadRequestError(
                f"{path}: threshold {name!r} must be a mapping with min and/or max"
            )
        out[name] = Threshold(min=spec.get("min"), max=spec.get("max"))
    return out


class GoldenRunner:
    def __init__(
        self,
        runner: AgentRunner,
        *,
        agent_name: str,
        agent_version: str,
        concurrency: int = 4,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._runner = runner
        self._name, self._version = agent_name, agent_version
        self._sem = asyncio.Semaphore(concurrency)
        self._timeout = timeout_seconds

    async def run(self, cases: Iterable[GoldenCase]) -> list[EvalRecord]:
        return list(await asyncio.gather(*(self._one(c) for c in cases)))

    async def _one(self, case: GoldenCase) -> EvalRecord:
        async with self._sem:
            request = AgentRequest(
                run_id=f"golden-{case.id}",
                thread_id=f"golden-{case.id}",
                principal=Principal(subject="golden-runner", tier="internal"),
                messages=(Message("user", case.input),),
                agent_name=self._name,
                agent_version=self._version,
            )
            output, tools, usage, error = "", [], Usage(), None
            started = time.perf_counter()
            try:
                async with asyncio.timeout(self._timeout):
                    async for ev in self._runner.stream(request):
                        if isinstance(ev, MessageCompleted):
                            output = ev.text
                        elif isinstance(ev, StepStarted) and ev.kind in (
                            StepKind.TOOL,
                            StepKind.MCP,
                        ):
                            tools.append(ev.name)
                        elif isinstance(ev, UsageReported):
                            usage = usage + ev.usage
                        elif isinstance(ev, RunError):
                            error = ev.code
            except DomainError as exc:  # same mapping ChatService applies to a live request
                error = exc.code
            except TimeoutError:
                error = "timeout"
            except Exception:
                error = "internal_error"
            return EvalRecord(
                case=case,
                output=output,
                tools_called=tuple(tools),
                error_code=error,
                latency_ms=(time.perf_counter() - started) * 1000,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=usage.cost_usd,
            )
