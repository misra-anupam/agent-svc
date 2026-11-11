"""LangSmith evaluator (extra: ``langsmith``): ``langsmith.evaluate`` over recorded outputs.

By default it runs locally (``upload_results=False``): no account or network. Set
``eval.langsmith_upload=true`` (and ``LANGSMITH_API_KEY``) to publish an experiment; that path
is not verified here (needs an account). Scores come from the shared scoring functions.
"""

from __future__ import annotations

import datetime as dt
import uuid
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
    tool_trajectory,
)


class LangSmithEvaluator:
    name = "langsmith"

    def __init__(self, cfg: EvalSettings) -> None:
        self._cfg = cfg

    def _run(
        self, records: Sequence[EvalRecord], agent_version: str, thresholds: dict[str, Threshold]
    ) -> EvalReport:
        from langsmith import Client, evaluate
        from langsmith.schemas import Example

        by_id = {r.case.id: r for r in records}
        now, dataset = dt.datetime.now(dt.UTC), uuid.uuid4()
        examples = [
            Example(
                id=uuid.uuid4(),
                dataset_id=dataset,
                created_at=now,
                inputs={"case_id": r.case.id, "question": r.case.input},
                outputs={"expected": r.case.expected},
            )
            for r in records
        ]

        def target(inputs: dict[str, Any]) -> dict[str, Any]:  # replay the recorded run
            return {"answer": by_id[inputs["case_id"]].output}

        def evaluator(fn: Any, key: str) -> Any:
            def run(run: Any, example: Any) -> dict[str, Any]:
                score = fn(by_id[example.inputs["case_id"]])
                return {"key": key, "score": score}

            run.__name__ = key
            return run

        upload = self._cfg.langsmith_upload
        result = evaluate(
            target,
            data=examples,
            evaluators=[
                evaluator(answer_match, "answer_match"),
                evaluator(tool_trajectory, "tool_trajectory"),
                evaluator(error_expectation, "error_expectation"),
            ],
            experiment_prefix=f"golden-{agent_version}",
            metadata={"agent_version": agent_version},
            upload_results=upload,
            client=Client() if upload else Client(api_key="local", auto_batch_tracing=False),
        )
        ls_scores: dict[str, list[float]] = {}
        for row in result:
            for res in row["evaluation_results"]["results"]:
                if res.score is not None:
                    ls_scores.setdefault(res.key, []).append(float(res.score))
        per_case = {r.case.id: case_scores(r) for r in records}
        return build_report(
            self.name,
            agent_version,
            records,
            per_case,
            thresholds,
            details={"langsmith_means": {k: sum(v) / len(v) for k, v in ls_scores.items()}},
        )

    async def evaluate(
        self,
        records: Sequence[EvalRecord],
        *,
        agent_version: str,
        thresholds: dict[str, Threshold],
    ) -> EvalReport:
        return await anyio.to_thread.run_sync(self._run, records, agent_version, thresholds)


def build(settings: Settings) -> LangSmithEvaluator:
    return LangSmithEvaluator(settings.evaluation)
