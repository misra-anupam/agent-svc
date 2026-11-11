from agent_svc.adapters.registries import build_runner
from agent_svc.config import Settings
from agent_svc.domain.models import AgentRequest, Message, MessageCompleted, Principal, RunCompleted


async def test_echo_streams_events_in_order() -> None:
    runner = build_runner(Settings(_env_file=None))  # type: ignore[call-arg]
    req = AgentRequest(
        run_id="r1",
        thread_id="t1",
        principal=Principal(subject="u"),
        messages=(Message("user", "hello world"),),
        agent_name="a",
        agent_version="1",
    )
    events = [e async for e in runner.stream(req)]
    assert events[0].type == "run.started"
    assert isinstance(events[-2], MessageCompleted) and events[-2].text == "hello world"
    assert isinstance(events[-1], RunCompleted)
