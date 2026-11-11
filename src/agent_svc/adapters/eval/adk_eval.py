"""Google ADK evaluation metrics (extra: ``adk_eval``), driven by our recorded runs.

* ``adk_tool_trajectory`` - ADK ``TrajectoryEvaluator`` (cases with ``expected_tools``; names only,
  arguments are not part of the golden schema).
* ``adk_response_match``  - ADK ``RougeEvaluator`` ROUGE-1 F1 against ``metadata.reference`` (cases
  that provide a full reference answer). ``expected`` is a substring, so it is not used here.

Works on any orchestrator: the recorded run is converted to ADK ``Invocation`` objects. The
deterministic ``answer_match`` / ``error_expectation`` metrics are reported alongside.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence

from agent_svc.config.settings import Settings
from agent_svc.domain.ports import EvalRecord, EvalReport, Threshold
from agent_svc.domain.scoring import build_report, case_scores


def _text(role: str, text: str):  # type: ignore[no-untyped-def]
    from google.genai import types

    return types.Content(role=role, parts=[types.Part(text=text)])


def _invocation(user: str, answer: str, tools: Sequence[str]):  # type: ignore[no-untyped-def]
    from google.adk.evaluation.eval_case import IntermediateData, Invocation
    from google.genai import types

    return Invocation(
        user_content=_text("user", user),
        final_response=_text("model", answer),
        intermediate_data=IntermediateData(
            tool_uses=[types.FunctionCall(name=t, args={}) for t in tools]
        ),
    )


async def _score(evaluator, actual, expected) -> float:  # type: ignore[no-untyped-def]
    res = evaluator.evaluate_invocations([actual], [expected])
    if inspect.isawaitable(res):
        res = await res
    return float(res.overall_score if res.overall_score is not None else 0.0)


class AdkEvaluator:
    name = "adk_eval"

    async def evaluate(
        self,
        records: Sequence[EvalRecord],
        *,
        agent_version: str,
        thresholds: dict[str, Threshold],
    ) -> EvalReport:
        from google.adk.evaluation.eval_metrics import EvalMetric
        from google.adk.evaluation.response_evaluator import RougeEvaluator
        from google.adk.evaluation.trajectory_evaluator import TrajectoryEvaluator

        trajectory = TrajectoryEvaluator(threshold=1.0)
        rouge = RougeEvaluator(EvalMetric(metric_name="response_match_score", threshold=0.0))
        per_case = {r.case.id: case_scores(r) for r in records}
        for r in records:
            scores = per_case[r.case.id]
            if r.case.expected_tools:
                scores["adk_tool_trajectory"] = await _score(
                    trajectory,
                    _invocation(r.case.input, r.output, r.tools_called),
                    _invocation(r.case.input, "", r.case.expected_tools),
                )
            if ref := r.case.metadata.get("reference"):
                scores["adk_response_match"] = await _score(
                    rouge,
                    _invocation(r.case.input, r.output, r.tools_called),
                    _invocation(r.case.input, str(ref), r.case.expected_tools),
                )
        return build_report(self.name, agent_version, records, per_case, thresholds)


def build(_settings: Settings) -> AdkEvaluator:
    return AdkEvaluator()
