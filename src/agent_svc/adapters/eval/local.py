"""Dependency-free evaluator: deterministic metrics only (no LLM judge, no network).

This is the CI gate: it needs no API keys, so a regression in tool routing, guardrails or the
answer contract fails the build.
"""

from __future__ import annotations

from collections.abc import Sequence

from agent_svc.config.settings import Settings
from agent_svc.domain.ports import EvalRecord, EvalReport, Threshold
from agent_svc.domain.scoring import build_report, case_scores


class LocalEvaluator:
    name = "local"

    async def evaluate(
        self,
        records: Sequence[EvalRecord],
        *,
        agent_version: str,
        thresholds: dict[str, Threshold],
    ) -> EvalReport:
        per_case = {r.case.id: case_scores(r) for r in records}
        return build_report(self.name, agent_version, records, per_case, thresholds)


def build(_settings: Settings) -> LocalEvaluator:
    return LocalEvaluator()
