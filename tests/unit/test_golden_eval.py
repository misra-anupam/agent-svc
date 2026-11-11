from __future__ import annotations

import argparse
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from tests.scripted import ScriptedModel, simulation_rules

from agent_svc.adapters import registries as reg
from agent_svc.adapters.llm.stack import build_llm_stack
from agent_svc.application.golden import GoldenRunner, load_cases, load_thresholds
from agent_svc.config.settings import Settings
from agent_svc.domain.exceptions import BadRequestError, PolicyViolationError
from agent_svc.domain.models import (
    AgentEvent,
    AgentRequest,
    MessageCompleted,
    RunCompleted,
    StepKind,
    StepStarted,
    Usage,
    UsageReported,
)
from agent_svc.domain.ports import EvalRecord, GoldenCase, Threshold
from agent_svc.domain.scoring import (
    answer_match,
    check_thresholds,
    operational_metrics,
    percentile,
    tool_trajectory,
)

C = GoldenCase


def rec(
    case: GoldenCase,
    output: str = "",
    tools: tuple[str, ...] = (),
    error: str | None = None,
    ms: float = 1.0,
) -> EvalRecord:
    return EvalRecord(case, output, tools, error, latency_ms=ms)


# ---- scoring ---------------------------------------------------------------------------------
def test_answer_match_is_case_and_whitespace_insensitive_and_optional() -> None:
    assert answer_match(rec(C("a", "q", expected="Paris"), "It is  PARIS\n.")) == 1.0
    assert answer_match(rec(C("a", "q", expected="Paris"), "Lyon")) == 0.0
    assert answer_match(rec(C("a", "q"), "anything")) is None


def test_tool_trajectory_is_exact_and_ordered() -> None:
    c = C("a", "q", expected_tools=("calculator", "current_time"))
    assert tool_trajectory(rec(c, tools=("calculator", "current_time"))) == 1.0
    assert tool_trajectory(rec(c, tools=("current_time", "calculator"))) == 0.0
    assert tool_trajectory(rec(c, tools=("calculator",))) == 0.0
    assert tool_trajectory(rec(C("a", "q"), tools=("x",))) is None  # no expectation: not scored


def test_expected_error_is_not_counted_as_an_unexpected_error() -> None:
    blocked = rec(C("b", "q", expect_error="policy_violation"), error="policy_violation")
    crashed = rec(C("c", "q"), error="internal_error")
    assert operational_metrics([blocked])["unexpected_error_rate"] == 0.0
    assert operational_metrics([blocked, crashed])["unexpected_error_rate"] == 0.5


def test_percentile_and_thresholds() -> None:
    assert percentile([10, 20, 30, 40], 0.95) == 40 and percentile([], 0.5) == 0.0
    fails = check_thresholds(
        {"a": 0.5, "lat": 900.0},
        {"a": Threshold(min=0.9), "lat": Threshold(max=500), "missing": Threshold(min=1)},
    )
    assert len(fails) == 3 and any("not computed" in f for f in fails)
    assert check_thresholds({"a": 1.0}, {"a": Threshold(min=1.0)}) == []


# ---- loading ---------------------------------------------------------------------------------
def write(tmp: Path, name: str, *rows: object) -> Path:
    p = tmp / name
    p.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows))
    return p


def test_load_cases_filters_by_tag_and_skips_comments(tmp_path: Path) -> None:
    write(
        tmp_path,
        "a.jsonl",
        "# comment",
        {"id": "1", "input": "x", "tags": ["core"]},
        {"id": "2", "input": "y", "tags": ["slow"]},
    )
    assert [c.id for c in load_cases(tmp_path)] == ["1", "2"]
    assert [c.id for c in load_cases(tmp_path, tags=["core"])] == ["1"]
    assert [c.id for c in load_cases(tmp_path, exclude_tags=["slow"])] == ["1"]


def test_bad_datasets_fail_loudly_with_file_and_line(tmp_path: Path) -> None:
    dup = tmp_path / "dup"
    dup.mkdir()
    write(dup, "dup.jsonl", {"id": "1", "input": "x"}, {"id": "1", "input": "y"})
    with pytest.raises(BadRequestError, match=r"dup.jsonl:2.*duplicate"):
        load_cases(dup)
    bad = write(tmp_path, "bad.jsonl", '{"id": "1"}')  # no "input"
    with pytest.raises(BadRequestError, match="invalid golden case"):
        load_cases(bad)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(BadRequestError, match="no golden dataset"):
        load_cases(empty)


def test_thresholds_yaml_is_validated(tmp_path: Path) -> None:
    ok = tmp_path / "t.yaml"
    ok.write_text("answer_match: {min: 0.9}\nlatency_p95_ms: {max: 100}\n")
    assert load_thresholds(ok) == {
        "answer_match": Threshold(min=0.9),
        "latency_p95_ms": Threshold(max=100),
    }
    bad = tmp_path / "bad.yaml"
    bad.write_text("answer_match: 0.9\n")
    with pytest.raises(BadRequestError):
        load_thresholds(bad)
    typo = tmp_path / "typo.yaml"
    typo.write_text("answer_match: {minimum: 0.9}\n")
    with pytest.raises(BadRequestError):
        load_thresholds(typo)


# ---- runner ----------------------------------------------------------------------------------
class Stub:
    async def stream(self, request: AgentRequest) -> AsyncIterator[AgentEvent]:
        rid, text = request.run_id, request.last_user_text
        if "block" in text:
            raise PolicyViolationError("nope")
        yield StepStarted(run_id=rid, step_id="s", kind=StepKind.TOOL, name="calculator")
        yield StepStarted(
            run_id=rid, step_id="g", kind=StepKind.GUARDRAIL, name="input"
        )  # not a tool
        yield UsageReported(run_id=rid, usage=Usage(3, 4, 0.5), model="m")
        yield MessageCompleted(run_id=rid, text=f"echo {text}")
        yield RunCompleted(run_id=rid)


async def test_runner_records_answers_tools_usage_and_domain_errors() -> None:
    cases = [C("ok", "hello"), C("blocked", "block me", expect_error="policy_violation")]
    recs = await GoldenRunner(Stub(), agent_name="a", agent_version="1").run(cases)
    ok, blocked = recs
    assert ok.output == "echo hello" and ok.tools_called == ("calculator",)
    assert (ok.input_tokens, ok.output_tokens, ok.cost_usd) == (3, 4, 0.5) and ok.error_code is None
    assert blocked.error_code == "policy_violation"


async def test_runner_turns_timeouts_into_an_error_code() -> None:
    class Hang:
        async def stream(self, request: AgentRequest) -> AsyncIterator[AgentEvent]:
            import asyncio

            await asyncio.sleep(5)
            yield RunCompleted(run_id="x")

    (r,) = await GoldenRunner(Hang(), agent_name="a", agent_version="1", timeout_seconds=0.05).run(
        [C("h", "x")]
    )
    assert r.error_code == "timeout"


# ---- every evaluator agrees, and every evaluator gates ----------------------------------------
RECORDS = [
    rec(
        C(
            "good",
            "q",
            expected="5",
            expected_tools=("calculator",),
            metadata={"reference": "It is 5"},
        ),
        "It is 5",
        ("calculator",),
    ),
    rec(C("bad", "q", expected="5", expected_tools=("calculator",)), "no idea", ()),
    rec(C("blocked", "q", expect_error="policy_violation"), "", (), "policy_violation"),
]
GATE = {"answer_match": Threshold(min=1.0), "tool_trajectory": Threshold(min=1.0)}


def evaluator_params() -> list[str]:
    names = ["mlflow_genai"]
    return names


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    s.evaluation.mlflow_tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    return s


@pytest.mark.parametrize("name", evaluator_params())
async def test_each_evaluator_reports_the_same_core_metrics_and_fails_the_gate(
    name: str, settings: Settings
) -> None:
    ev = reg.evaluators.create(name, settings)
    report = await ev.evaluate(RECORDS, agent_version="9.9.9", thresholds=GATE)
    assert report.agent_version == "9.9.9" and report.evaluator == name
    assert report.metrics["answer_match"] == 0.5 and report.metrics["tool_trajectory"] == 0.5
    assert report.metrics["error_expectation"] == 1.0  # the block was expected; nothing crashed
    assert not report.passed and len(report.failures) == 2
    ok = await ev.evaluate(RECORDS[:1], agent_version="9.9.9", thresholds=GATE)
    assert ok.passed and ok.failures == ()


async def test_mlflow_run_is_tagged_with_agent_version_and_gate_result(settings: Settings) -> None:
    import mlflow

    ev = reg.evaluators.create("mlflow_genai", settings)
    report = await ev.evaluate(RECORDS, agent_version="2.3.4", thresholds=GATE)
    run = mlflow.get_run(report.details["mlflow_run_id"])
    assert run.data.tags["agent.version"] == "2.3.4" and run.data.tags["gate.passed"] == "false"
    assert run.data.metrics["gate/answer_match"] == 0.5


# ---- the CI gate itself: exit code follows the thresholds -------------------------------------
@pytest.fixture(autouse=True)
def _sqlite_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    monkeypatch.setenv("AGENT_EVALUATION__MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/ml.db")


def scripted(monkeypatch: pytest.MonkeyPatch, dataset: str) -> None:
    """Serve the dataset's ``metadata.simulate`` replies through the real LLM stack."""
    rules = simulation_rules(load_cases(dataset, tags=[], exclude_tags=[]))
    monkeypatch.setattr(
        reg,
        "build_llm",
        lambda s: build_llm_stack(s.llm, lambda m: ScriptedModel(m, rules=rules)),
    )


def cli_args(tmp: Path, thresholds: Path | None = None, **kw: object) -> argparse.Namespace:
    base = dict(
        evaluator="mlflow_genai",
        orchestrator=None,
        dataset=None,
        thresholds=str(thresholds) if thresholds else None,
        tags=[],
        exclude_tags=["none"],  # include simulate-only cases: the scripted model serves them
        concurrency=4,
        report_dir=str(tmp / "reports"),
    )
    return argparse.Namespace(**{**base, **kw})


async def test_shipped_dataset_passes_the_shipped_thresholds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_svc.cli.golden import run

    scripted(monkeypatch, "evals/golden")
    assert await run(cli_args(tmp_path)) == 0
    (report,) = (tmp_path / "reports").glob("*.json")
    assert json.loads(report.read_text())["passed"] is True


async def test_gate_exits_nonzero_when_a_threshold_is_violated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_svc.cli.golden import run

    scripted(monkeypatch, "evals/golden")
    strict = tmp_path / "strict.yaml"
    strict.write_text("latency_p95_ms: {max: 0.0001}\n")  # impossible: must fail
    assert await run(cli_args(tmp_path, strict)) == 1


async def test_a_regression_in_agent_behaviour_fails_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the model stops calling the calculator (a routing regression), the gate catches it."""
    from agent_svc.cli.golden import run

    data = tmp_path / "ds"
    data.mkdir()
    write(
        data,
        "r.jsonl",
        {
            "id": "calc",
            "input": "What is 12 * 7?",
            "expected": "84",
            "expected_tools": ["calculator"],
            "metadata": {"simulate": {"reply": "84"}},
        },
    )  # scripted model never calls the tool
    thresholds = tmp_path / "t.yaml"
    thresholds.write_text("tool_trajectory: {min: 1.0}\n")
    scripted(monkeypatch, str(data))
    assert await run(cli_args(tmp_path, thresholds, dataset=str(data))) == 1
