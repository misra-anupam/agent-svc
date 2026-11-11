"""Dependency-free reference runner: echoes the last user message. Used for tests and demos."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

from agent_svc.config.settings import Settings
from agent_svc.domain.models import (
    AgentEvent,
    AgentRequest,
    MessageCompleted,
    MessageDelta,
    RunCompleted,
    RunStarted,
)


class EchoRunner:
    async def stream(self, request: AgentRequest) -> AsyncIterator[AgentEvent]:
        yield RunStarted(
            run_id=request.run_id,
            thread_id=request.thread_id,
            agent_name=request.agent_name,
            agent_version=request.agent_version,
        )
        text = request.last_user_text
        for piece in re.findall(r"\S+\s*", text):
            yield MessageDelta(run_id=request.run_id, text=piece)
        yield MessageCompleted(run_id=request.run_id, text=text)
        yield RunCompleted(run_id=request.run_id)


def build_echo(_settings: Settings, _llm: object = None, _tools: object = None) -> EchoRunner:
    return EchoRunner()
