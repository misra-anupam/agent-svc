"""ChatService: the use case behind the SSE endpoint. Depends only on ports."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Sequence
from typing import Any

import structlog

from agent_svc.application.admission import Admission, AdmissionService
from agent_svc.application.memory_facade import MemoryFacade
from agent_svc.domain.exceptions import DomainError
from agent_svc.domain.models import (
    AgentEvent,
    AgentRequest,
    Message,
    MessageCompleted,
    Principal,
    RunCompleted,
    RunError,
)
from agent_svc.domain.ports import AgentRunner

log = structlog.get_logger()


class ChatService:
    def __init__(
        self,
        runner: AgentRunner,
        *,
        agent_name: str,
        agent_version: str,
        memory: MemoryFacade | None = None,
        admission: AdmissionService | None = None,
    ) -> None:
        self._runner = runner
        self._agent_name = agent_name
        self._agent_version = agent_version
        self._memory = memory
        self._admission = admission

    async def admit(self, principal: Principal) -> Admission:
        """Raises 429/503 *before* the HTTP stream starts, so clients get proper status codes."""
        return await self._admission.admit(principal) if self._admission else Admission()

    async def release(self, admission: Admission) -> None:
        if self._admission:
            await self._admission.release(admission)

    async def prepare(
        self,
        thread_id: str,
        principal: Principal,
        new_messages: Sequence[Message],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> AgentRequest:
        context = (
            await self._memory.build_context(principal, thread_id, new_messages)
            if self._memory
            else list(new_messages)
        )
        return AgentRequest(
            run_id=uuid.uuid4().hex,
            thread_id=thread_id,
            principal=principal,
            messages=tuple(context),
            agent_name=self._agent_name,
            agent_version=self._agent_version,
            metadata=metadata or {},
        )

    async def stream(
        self,
        request: AgentRequest,
        *,
        admission: Admission | None = None,
        new_messages: Sequence[Message] = (),
    ) -> AsyncIterator[AgentEvent]:
        """Yields events; failures become a terminal ``run.error`` event (never raises)."""
        answer = ""
        try:
            async for event in self._runner.stream(request):
                if isinstance(event, MessageCompleted):
                    answer = event.text
                elif isinstance(event, RunCompleted):
                    await self._after_success(request, new_messages, answer, event)
                yield event
        except DomainError as exc:
            log.warning("run_failed", run_id=request.run_id, code=exc.code, detail=exc.detail)
            yield RunError(
                run_id=request.run_id, code=exc.code, message=exc.detail, retryable=exc.retryable
            )
        except Exception:
            log.exception("run_crashed", run_id=request.run_id)
            yield RunError(run_id=request.run_id, code="internal_error", message="Internal error")
        finally:
            if admission is not None:
                await self.release(admission)

    async def _after_success(
        self,
        request: AgentRequest,
        new_messages: Sequence[Message],
        answer: str,
        done: RunCompleted,
    ) -> None:
        if self._admission:
            await self._admission.record_usage(request.principal, done.usage)
        if self._memory and new_messages and answer:
            await self._memory.save_turn(request.principal, request.thread_id, new_messages, answer)
