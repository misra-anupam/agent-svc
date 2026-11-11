from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from agent_svc.domain.models import AgentEvent, AgentRequest


class AgentRunner(Protocol):
    """Runs one agent turn and streams framework-neutral events.

    Implementations wrap LangGraph/ADK/deepagents/MAF/CrewAI. Decorators (guardrails, cache,
    tracing, metering) implement the same port, so they stack freely.
    """

    def stream(self, request: AgentRequest) -> AsyncIterator[AgentEvent]: ...
