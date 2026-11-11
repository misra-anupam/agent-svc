"""ChatService: the use case behind the SSE endpoint. Depends only on ports."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Sequence

import structlog

from agent_svc.domain.exceptions import DomainError
from agent_svc.domain.models import (
    AgentEvent,
    AgentRequest,
    Message,
    Principal,
    RunError,
)
from agent_svc.domain.ports import AgentRunner

log = structlog.get_logger()


class ChatService:
    def __init__(self, runner: AgentRunner, *, agent_name: str, agent_version: str) -> None:
        self._runner = runner
        self._agent_name = agent_name
        self._agent_version = agent_version

    def new_request(
        self, thread_id: str, principal: Principal, messages: Sequence[Message]
    ) -> AgentRequest:
        return AgentRequest(
            run_id=uuid.uuid4().hex,
            thread_id=thread_id,
            principal=principal,
            messages=tuple(messages),
            agent_name=self._agent_name,
            agent_version=self._agent_version,
        )

    async def stream(self, request: AgentRequest) -> AsyncIterator[AgentEvent]:
        """Yields events; any failure becomes a terminal ``run.error`` event (never raises)."""
        try:
            async for event in self._runner.stream(request):
                yield event
        except DomainError as exc:
            log.warning("run_failed", run_id=request.run_id, code=exc.code, detail=exc.detail)
            yield RunError(
                run_id=request.run_id, code=exc.code, message=exc.detail, retryable=exc.retryable
            )
        except Exception:
            log.exception("run_crashed", run_id=request.run_id)
            yield RunError(run_id=request.run_id, code="internal_error", message="Internal error")
