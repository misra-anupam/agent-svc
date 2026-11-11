"""Pure scoring functions shared by every evaluator, so metric names mean the same everywhere."""

from __future__ import annotations

import math
from collections.abc import Sequence

from agent_svc.domain.ports.evaluation import EvalRecord, EvalReport, Threshold


def answer_match(r: EvalRecord) -> float | None:
    """1.0 if the expected text is in the answer (None when the case has no expectation)."""
    if r.case.expected is None:
        return None
    return float(r.case.expected.casefold() in " ".join(r.output.casefold().split()))


def tool_trajectory(r: EvalRecord) -> float | None:
    """1.0 if the tools called match the expected ordered trajectory exactly."""
    if not r.case.expected_tools:
        return None
    return float(tuple(r.tools_called) == tuple(r.case.expected_tools))


def error_expectation(r: EvalRecord) -> float:
    """1.0 when the run ended as expected: no error, or exactly the expected error code."""
    return float(r.error_code == r.case.expect_error)


def case_scores(r: EvalRecord) -> dict[str, float]:
    scores = {"error_expectation": error_expectation(r)}
    if (m := answer_match(r)) is not None:
        scores["answer_match"] = m
    if (t := tool_trajectory(r)) is not None:
        scores["tool_trajectory"] = t
    return scores


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]


def operational_metrics(records: Sequence[EvalRecord]) -> dict[str, float]:
    n = len(records) or 1
    lat = [r.latency_ms for r in records]
    return {
        "unexpected_error_rate": sum(r.error_code != r.case.expect_error for r in records) / n,
        "latency_p50_ms": percentile(lat, 0.5),
        "latency_p95_ms": percentile(lat, 0.95),
        "cost_usd_total": sum(r.cost_usd for r in records),
        "tokens_mean": sum(r.input_tokens + r.output_tokens for r in records) / n,
    }


def aggregate(per_case: dict[str, dict[str, float]]) -> dict[str, float]:
    """Mean per metric over the cases it applies to."""
    buckets: dict[str, list[float]] = {}
    for scores in per_case.values():
        for k, v in scores.items():
            buckets.setdefault(k, []).append(v)
    return {k: sum(v) / len(v) for k, v in buckets.items()}


def check_thresholds(metrics: dict[str, float], thresholds: dict[str, Threshold]) -> list[str]:
    """Human-readable failures. A thresholded metric that was never computed is a failure."""
    failures: list[str] = []
    for name, t in thresholds.items():
        if name not in metrics:
            failures.append(f"{name}: not computed (no applicable cases?)")
            continue
        v = metrics[name]
        if t.min is not None and v < t.min:
            failures.append(f"{name}: {v:.4g} < min {t.min:g}")
        if t.max is not None and v > t.max:
            failures.append(f"{name}: {v:.4g} > max {t.max:g}")
    return failures


def build_report(
    evaluator: str,
    agent_version: str,
    records: Sequence[EvalRecord],
    per_case: dict[str, dict[str, float]],
    thresholds: dict[str, Threshold],
    *,
    extra_metrics: dict[str, float] | None = None,
    details: dict[str, object] | None = None,
) -> EvalReport:
    metrics = {**aggregate(per_case), **operational_metrics(records), **(extra_metrics or {})}
    failures = check_thresholds(metrics, thresholds)
    return EvalReport(
        evaluator=evaluator,
        agent_version=agent_version,
        metrics=metrics,
        passed=not failures,
        failures=tuple(failures),
        per_case=per_case,
        details=dict(details or {}),
    )
