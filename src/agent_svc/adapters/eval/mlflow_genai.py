"""MLflow GenAI evaluator (extra: ``mlflow_genai``): ``mlflow.genai.evaluate`` + run tracking.

The deterministic scorers wrap the shared scoring functions (same numbers as ``local``). The run
is tagged with the agent version and carries the operational metrics too. Set
``eval.judge_model`` to add MLflow's ``Correctness`` LLM judge (needs that model's credentials;
not exercised offline). Thresholds are always checked on our own aggregates.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import anyio

from agent_svc.config.settings import EvalSettings, Settings
from agent_svc.domain.ports import EvalRecord, EvalReport, Threshold
from agent_svc.domain.scoring import (
    answer_match,
    build_report,
    case_scores,
    error_expectation,
    operational_metrics,
    tool_trajectory,
)


class MlflowGenAIEvaluator:
    name = "mlflow_genai"

    def __init__(self, cfg: EvalSettings) -> None:
        self._cfg = cfg

    def _run(
        self, records: Sequence[EvalRecord], agent_version: str, thresholds: dict[str, Threshold]
    ) -> EvalReport:
        import mlflow
        from mlflow.genai.scorers import scorer

        mlflow.set_tracking_uri(
            self._cfg.mlflow_tracking_uri or "sqlite:///evals/.mlflow/mlflow.db"
        )
        mlflow.set_experiment(self._cfg.mlflow_experiment)
        by_id = {r.case.id: r for r in records}

        @scorer
        def answer_match_s(inputs: dict[str, Any]) -> float | None:
            return answer_match(by_id[inputs["case_id"]])

        @scorer
        def tool_trajectory_s(inputs: dict[str, Any]) -> float | None:
            return tool_trajectory(by_id[inputs["case_id"]])

        @scorer
        def error_expectation_s(inputs: dict[str, Any]) -> float:
            return error_expectation(by_id[inputs["case_id"]])

        scorers: list[Any] = [answer_match_s, tool_trajectory_s, error_expectation_s]
        if self._cfg.judge_model:
            from mlflow.genai.scorers import Correctness

            scorers.append(Correctness(model=self._cfg.judge_model))

        data = [
            {
                "inputs": {"case_id": r.case.id, "question": r.case.input},
                "outputs": r.output,
                "expectations": {"expected_response": r.case.expected} if r.case.expected else {},
            }
            for r in records
        ]
        per_case = {r.case.id: case_scores(r) for r in records}
        with mlflow.start_run(run_name=f"golden-{agent_version}") as run:
            mlflow.set_tags({"agent.version": agent_version, "evaluator": self.name})
            result = mlflow.genai.evaluate(data=data, scorers=scorers)
            mlflow.log_metrics(operational_metrics(records))
            mlflow_metrics = {k: float(v) for k, v in result.metrics.items()}
            report = build_report(
                self.name,
                agent_version,
                records,
                per_case,
                thresholds,
                extra_metrics={
                    k.removesuffix("/mean"): v
                    for k, v in mlflow_metrics.items()
                    if k.startswith("correctness")
                },
                details={"mlflow_run_id": run.info.run_id, "mlflow_metrics": mlflow_metrics},
            )
            mlflow.log_metrics({f"gate/{k}": v for k, v in report.metrics.items()})
            mlflow.set_tag("gate.passed", str(report.passed).lower())
        return report

    async def evaluate(
        self,
        records: Sequence[EvalRecord],
        *,
        agent_version: str,
        thresholds: dict[str, Threshold],
    ) -> EvalReport:
        return await anyio.to_thread.run_sync(self._run, records, agent_version, thresholds)


def build(settings: Settings) -> MlflowGenAIEvaluator:
    return MlflowGenAIEvaluator(settings.evaluation)
