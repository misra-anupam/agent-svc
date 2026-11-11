"""Every orchestrator adapter must emit the same event grammar. Add new adapters to PROVIDERS."""

import pytest
from tests.helpers import make_request

from agent_svc.adapters.llm.fake import FakeLLM
from agent_svc.adapters.registries import orchestrators
from agent_svc.config import Settings
from agent_svc.domain.models import (
    MessageCompleted,
    MessageDelta,
    RunCompleted,
    RunStarted,
    StepFinished,
    StepStarted,
)

PROVIDERS = ["echo", "langgraph"]


@pytest.fixture(params=PROVIDERS)
def runner(request: pytest.FixtureRequest):  # type: ignore[no-untyped-def]
    llm = FakeLLM(reply="alpha beta gamma")
    return orchestrators.create(request.param, Settings(_env_file=None), llm)  # type: ignore[call-arg]


async def test_event_grammar(runner) -> None:  # type: ignore[no-untyped-def]
    events = [e async for e in runner.stream(make_request("alpha beta gamma", run_id="r9"))]
    assert isinstance(events[0], RunStarted) and isinstance(events[-1], RunCompleted)
    assert all(e.run_id == "r9" for e in events)
    deltas = "".join(e.text for e in events if isinstance(e, MessageDelta))
    done = [e for e in events if isinstance(e, MessageCompleted)]
    assert len(done) == 1 and deltas == done[0].text and deltas.strip() == "alpha beta gamma"


async def test_steps_are_balanced_and_parents_exist(runner) -> None:  # type: ignore[no-untyped-def]
    events = [e async for e in runner.stream(make_request())]
    started = {e.step_id for e in events if isinstance(e, StepStarted)}
    finished = {e.step_id for e in events if isinstance(e, StepFinished)}
    assert started == finished
    for e in events:
        if isinstance(e, StepStarted) and e.parent_id:
            assert e.parent_id in started
