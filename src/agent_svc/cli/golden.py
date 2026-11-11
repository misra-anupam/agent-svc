"""``agent-svc-eval``: replay the golden dataset through the service stack and gate on thresholds.

    uv run agent-svc-eval                          # your configured model (OPENROUTER_API_KEY)
    uv run agent-svc-eval --tags guardrails

Exit code 1 when any threshold is violated, so CI can gate on it. Cases tagged ``simulate-only``
assume scripted replies and are skipped unless you pass ``--exclude-tags`` yourself.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
import time
from pathlib import Path

from agent_svc.adapters import registries as reg
from agent_svc.application.golden import GoldenRunner, load_cases, load_thresholds
from agent_svc.config.settings import Settings
from agent_svc.container import Container
from agent_svc.domain.ports import EvalReport, Threshold


def _configure(settings: Settings, *, orchestrator: str | None) -> None:
    """Harness overrides: no infra, no auth or limits, a fixed tool set, no answer cache."""
    settings.cache.provider = "none"  # a cached answer would hide a regression
    if orchestrator:
        settings.orchestrator.provider = orchestrator  # type: ignore[assignment]
    settings.tools.local = ["calculator", "current_time"]
    settings.redis.enabled = settings.postgres.enabled = False
    settings.ratelimit.provider = "none"
    settings.auth.enabled = False
    if not settings.tracing.mlflow_tracking_uri.startswith(("sqlite", "file")):
        settings.tracing.mlflow_tracking_uri = settings.evaluation.mlflow_tracking_uri or (
            "sqlite:///evals/.mlflow/mlflow.db"
        )
    settings.mcp.servers = []
    settings.memory.thread_provider = settings.memory.user_provider = "none"
    settings.memory.long_term_provider = "none"


def render(report: EvalReport, thresholds: dict[str, Threshold]) -> str:
    lines = [f"evaluator={report.evaluator} agent_version={report.agent_version}", ""]
    lines.append(f"{'metric':<24}{'value':>12}  gate")
    for name, value in sorted(report.metrics.items()):
        t = thresholds.get(name)
        gate = ""
        if t:
            bad = any(f.startswith(f"{name}:") for f in report.failures)
            gate = (
                ("FAIL " if bad else "ok   ")
                + (f"min {t.min:g} " if t.min is not None else "")
                + (f"max {t.max:g}" if t.max is not None else "")
            )
        lines.append(f"{name:<24}{value:>12.4g}  {gate}")
    failing = sorted(c for c, s in report.per_case.items() if any(v < 1.0 for v in s.values()))
    if failing:
        lines += ["", "cases with a score below 1.0: " + ", ".join(failing)]
    lines += ["", "PASSED" if report.passed else "FAILED: " + "; ".join(report.failures)]
    return "\n".join(lines)


def _write_report(report: EvalReport, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    path = out / f"{report.agent_version}-{report.evaluator}-{stamp}.json"
    path.write_text(json.dumps(dataclasses.asdict(report), indent=2, default=str))
    return path


async def run(args: argparse.Namespace) -> int:
    settings = Settings()
    cases = load_cases(
        args.dataset or settings.evaluation.golden_dir,
        tags=args.tags,
        exclude_tags=args.exclude_tags or ("simulate-only",),
    )
    if not cases:
        print("no golden cases selected", file=sys.stderr)
        return 2
    thresholds = load_thresholds(args.thresholds or settings.evaluation.thresholds_file)
    _configure(settings, orchestrator=args.orchestrator)
    evaluator = reg.evaluators.create(args.evaluator or settings.evaluation.provider, settings)

    container = Container(settings)
    await container.start()
    try:
        assert container.runner is not None
        records = await GoldenRunner(
            container.runner,
            agent_name=container.settings.app.agent_name,
            agent_version=container.settings.app.agent_version,
            concurrency=args.concurrency,
        ).run(cases)
    finally:
        await container.stop()
    report = await evaluator.evaluate(
        records, agent_version=container.settings.app.agent_version, thresholds=thresholds
    )
    print(render(report, thresholds))
    print(f"\nreport: {_write_report(report, Path(args.report_dir))}")
    return 0 if report.passed else 1


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="agent-svc-eval", description=__doc__)
    p.add_argument(
        "--evaluator", choices=sorted(reg.evaluators.names()), help="default: eval.provider"
    )
    p.add_argument("--orchestrator", help="override orchestrator.provider")
    p.add_argument("--dataset", help="file or directory (default: eval.golden_dir)")
    p.add_argument("--thresholds", help="YAML (default: eval.thresholds_file)")
    p.add_argument("--tags", nargs="*", default=[])
    p.add_argument("--exclude-tags", nargs="*", default=[])
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--report-dir", default="evals/reports")
    sys.exit(asyncio.run(run(p.parse_args(argv))))


if __name__ == "__main__":
    main()
