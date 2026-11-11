import sys
from pathlib import Path

import pytest
from tests.helpers import parse_sse

from agent_svc.adapters.llm.fake import FakeLLM
from agent_svc.config.settings import McpServerConfig
from agent_svc.domain.models import ToolCall

SERVER = str(Path(__file__).parent.parent / "fixtures" / "mcp_server.py")
BODY = {"messages": [{"role": "user", "content": "add 2 and 3 with the demo server"}]}


@pytest.mark.parametrize("orchestrator", ["langgraph"])
async def test_agent_calls_mcp_tool_and_ui_sees_an_mcp_step(
    settings,
    make_client,
    orchestrator: str,  # type: ignore[no-untyped-def]
) -> None:
    settings.orchestrator.provider = orchestrator
    settings.mcp.servers = [
        McpServerConfig(name="demo", transport="stdio", command=[sys.executable, SERVER, "stdio"])
    ]
    settings.mcp.connect_timeout_seconds = 30
    primary = FakeLLM(
        "fake/primary",
        reply="The sum is 5",
        tool_plan=[ToolCall("c1", "demo__add", {"a": 2, "b": 3})],
    )
    async with make_client({"fake/primary": primary, "fake/secondary": FakeLLM()}) as c:
        r = await c.post("/v1/threads/t/chat", json=BODY)
    events = parse_sse(r.text)
    started = next(d for t, d in events if t == "step.started" and d["name"] == "demo__add")
    finished = next(
        d for t, d in events if t == "step.finished" and d["step_id"] == started["step_id"]
    )
    assert started["kind"] == "mcp" and finished["ok"] is True and finished["output_summary"] == "5"
    assert events[-1][0] == "run.completed"
    assert next(d for t, d in events if t == "message.completed")["text"].strip() == "The sum is 5"


async def test_disallowed_mcp_host_fails_startup_not_requests(settings, make_client) -> None:  # type: ignore[no-untyped-def]
    from agent_svc.domain.exceptions import ConfigurationError

    settings.mcp.servers = [
        McpServerConfig(name="x", transport="streamable_http", url="http://169.254.169.254/mcp")
    ]
    with pytest.raises(ConfigurationError, match="metadata"):
        async with make_client():
            pass
