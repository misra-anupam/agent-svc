"""Every orchestrator adapter must emit the same event grammar for the same scenarios.

Add a new adapter by appending its registry name to PROVIDERS; nothing else changes.
"""

import os

import pytest
from tests.helpers import make_request

from agent_svc.adapters.llm.fake import FakeLLM
from agent_svc.adapters.registries import orchestrators
from agent_svc.adapters.tools.composite import GuardedToolProvider
from agent_svc.adapters.tools.local import BUILTIN, LocalToolProvider
from agent_svc.config import Settings
from agent_svc.domain.exceptions import IterationLimitError
from agent_svc.domain.models import (
    AgentEvent,
    MessageCompleted,
    MessageDelta,
    RunCompleted,
    RunStarted,
    StepFinished,
    StepKind,
    StepStarted,
    ToolCall,
    UsageReported,
)

# CI runs each extra in isolation: CONTRACT_PROVIDERS=adk pytest tests/contract
PROVIDERS = os.environ.get("CONTRACT_PROVIDERS", "langgraph,deepagents,adk,maf,crewai").split(",")
CALC = ToolCall("c1", "calculator", {"expression": "2+3"})


def make_runner(name: str, llm: FakeLLM, *, max_iterations: int = 8):  # type: ignore[no-untyped-def]
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    settings.orchestrator.max_iterations = max_iterations
    tools = GuardedToolProvider(
        LocalToolProvider([BUILTIN["calculator"]]),
        allowlist=None,
        timeout_seconds=5,
        max_result_chars=1000,
    )
    return orchestrators.create(name, settings, llm, tools)


async def collect(runner, text: str = "what is 2+3") -> list[AgentEvent]:  # type: ignore[no-untyped-def]
    return [e async for e in runner.stream(make_request(text, run_id="r9"))]


def tool_shape(events: list[AgentEvent]) -> list[str]:
    """Framework-neutral skeleton: tool/MCP/RAG steps only (extra agent steps are allowed)."""
    tool_ids = {
        e.step_id: e for e in events if isinstance(e, StepStarted) and e.kind != StepKind.AGENT
    }
    out: list[str] = []
    for e in events:
        if isinstance(e, StepStarted) and e.step_id in tool_ids:
            out.append(f"start:{e.kind.value}:{e.name}")
        elif isinstance(e, StepFinished) and e.step_id in tool_ids:
            out.append("finish:ok" if e.ok else "finish:error")
    return out


@pytest.fixture(params=PROVIDERS)
def provider(request: pytest.FixtureRequest) -> str:
    return str(request.param)


async def test_plain_answer_event_grammar(provider: str) -> None:
    events = await collect(make_runner(provider, FakeLLM(reply="alpha beta gamma")), "hello")
    assert isinstance(events[0], RunStarted) and isinstance(events[-1], RunCompleted)
    assert all(e.run_id == "r9" for e in events)
    deltas = "".join(e.text for e in events if isinstance(e, MessageDelta))
    done = [e for e in events if isinstance(e, MessageCompleted)]
    assert len(done) == 1 and deltas == done[0].text and deltas.strip() == "alpha beta gamma"
    assert any(isinstance(e, UsageReported) and e.usage.total_tokens > 0 for e in events)
    assert tool_shape(events) == []


async def test_tool_call_produces_balanced_nested_steps(provider: str) -> None:
    llm = FakeLLM(reply="The answer is 5", tool_plan=[CALC])
    events = await collect(make_runner(provider, llm))
    assert tool_shape(events) == ["start:tool:calculator", "finish:ok"]
    started = {e.step_id for e in events if isinstance(e, StepStarted)}
    assert started == {e.step_id for e in events if isinstance(e, StepFinished)}
    assert all(e.parent_id in started for e in events if isinstance(e, StepStarted) and e.parent_id)
    tool_end = next(e for e in events if isinstance(e, StepFinished))
    assert "5" in tool_end.output_summary  # the real calculator ran

    # the answer is produced after the tool, and only the final text is streamed
    first_delta = next(i for i, e in enumerate(events) if isinstance(e, MessageDelta))
    assert events.index(tool_end) < first_delta
    done = next(e for e in events if isinstance(e, MessageCompleted))
    assert done.text.strip() == "The answer is 5"
    assert sum(isinstance(e, UsageReported) for e in events) >= 2  # one per model call
    # the tool result really reached the model on its second call
    assert any(m.role == "tool" and "5" in m.content for m in llm.seen[-1])


async def test_failing_tool_is_reported_and_agent_recovers(provider: str) -> None:
    bad = ToolCall("c1", "calculator", {"expression": "1/0"})
    llm = FakeLLM(reply="could not compute", tool_plan=[bad])
    events = await collect(make_runner(provider, llm), "divide")
    assert tool_shape(events) == ["start:tool:calculator", "finish:error"]
    assert (
        next(e for e in events if isinstance(e, MessageCompleted)).text.strip()
        == "could not compute"
    )
    assert any(m.role == "tool" and "division by zero" in m.content for m in llm.seen[-1])


async def test_invalid_tool_arguments_are_rejected_before_execution(provider: str) -> None:
    bad = ToolCall("c1", "calculator", {"wrong": 1})  # schema requires "expression"
    llm = FakeLLM(reply="ok", tool_plan=[bad])
    events = await collect(make_runner(provider, llm), "x")
    assert tool_shape(events) == ["start:tool:calculator", "finish:error"]
    # wording differs per framework (some validate before our guard); the model must be told
    assert any(m.role == "tool" and "argument" in m.content.lower() for m in llm.seen[-1])


async def test_runaway_tool_loop_is_stopped(provider: str) -> None:
    llm = FakeLLM(reply="never reached", tool_plan=[CALC] * 40)
    with pytest.raises(IterationLimitError):
        await collect(make_runner(provider, llm, max_iterations=2), "loop forever")
    assert llm.calls < 40
