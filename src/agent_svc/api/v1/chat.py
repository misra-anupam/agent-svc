from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, Path, Request
from fastapi.responses import StreamingResponse
from pydantic import TypeAdapter

from agent_svc.api.deps import get_chat_service, get_kv, get_principal, get_supervisor
from agent_svc.api.schemas.chat import ChatRequest, SseEvent
from agent_svc.api.sse import RunSupervisor
from agent_svc.application.chat_service import ChatService
from agent_svc.domain.exceptions import (
    BadRequestError,
    NotFoundError,
    PayloadTooLargeError,
    UnavailableError,
)
from agent_svc.domain.models import Message, Principal
from agent_svc.domain.ports import Cache

router = APIRouter(tags=["chat"])
_ID = r"^[A-Za-z0-9_-]{1,64}$"
_IDEM = r"^[A-Za-z0-9_.:-]{8,128}$"
_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
_SSE_DOC: dict[int | str, dict[str, Any]] = {
    200: {
        "content": {"text/event-stream": {"schema": TypeAdapter(SseEvent).json_schema()}},
        "description": "Stream of typed events: run.started, step.started/finished "
        "(tool/rag/agent/mcp/guardrail, nested via parent_id), message.delta, usage, "
        "message.completed, run.error, run.completed. Each frame has a numeric `id`.",
    }
}


def _stream(sup: RunSupervisor, run_id: str, after: int, heartbeat: float) -> StreamingResponse:
    return StreamingResponse(
        sup.tail(run_id, after=after, heartbeat_seconds=heartbeat),
        media_type="text/event-stream",
        headers={**_SSE_HEADERS, "X-Run-ID": run_id},
    )


@router.post(
    "/threads/{thread_id}/chat",
    summary="Chat with the agent (Server-Sent Events)",
    response_class=StreamingResponse,
    responses=_SSE_DOC,
)
async def chat(
    request: Request,
    body: ChatRequest,
    thread_id: Annotated[str, Path(pattern=_ID)],
    principal: Annotated[Principal, Depends(get_principal)],
    service: Annotated[ChatService, Depends(get_chat_service)],
    supervisor: Annotated[RunSupervisor, Depends(get_supervisor)],
    kv: Annotated[Cache | None, Depends(get_kv)],
    idempotency_key: Annotated[str | None, Header(pattern=_IDEM)] = None,
    cache_control: Annotated[str | None, Header()] = None,
) -> StreamingResponse:
    cfg = request.app.state.settings.app
    if len(body.messages) > cfg.max_messages_per_request:
        raise BadRequestError(f"at most {cfg.max_messages_per_request} messages per request")
    if any(len(m.content) > cfg.max_message_chars for m in body.messages):
        raise PayloadTooLargeError(f"message exceeds {cfg.max_message_chars} characters")
    if body.messages[-1].role != "user":
        raise BadRequestError("the last message must have role 'user'")
    new = [Message(m.role, m.content) for m in body.messages]

    # Idempotent replay happens *before* admission so retries never consume quota.
    store: Cache | None = None
    if idempotency_key:
        if kv is None:
            raise UnavailableError("Idempotency-Key needs Redis, which is not enabled")
        store = kv
    idem_key = f"idem:{principal.tenant}:{principal.subject}:{thread_id}:{idempotency_key}"
    if store is not None:
        prior = await store.get(idem_key)
        if prior:
            run_id = prior.decode()
            if await supervisor.owner(run_id) == principal.subject:
                return _stream(supervisor, run_id, 0, cfg.sse_heartbeat_seconds)

    admission = await service.admit(principal)  # 429/503 as proper problem+json responses
    try:
        metadata = {"no_cache": "no-cache" in (cache_control or "").lower()}
        areq = await service.prepare(thread_id, principal, new, metadata=metadata)
        if store is not None and not await store.set_if_absent(
            idem_key, areq.run_id.encode(), 3600
        ):
            prior = await store.get(idem_key)  # lost a race with a concurrent identical request
            if prior and await supervisor.owner(prior.decode()) == principal.subject:
                await service.release(admission)
                return _stream(supervisor, prior.decode(), 0, cfg.sse_heartbeat_seconds)
        await supervisor.start(
            areq.run_id,
            principal.subject,
            service.stream(areq, admission=admission, new_messages=new),
        )
    except BaseException:
        await service.release(admission)
        raise
    return _stream(supervisor, areq.run_id, 0, cfg.sse_heartbeat_seconds)


@router.get(
    "/runs/{run_id}/events",
    summary="Resume or replay a run's event stream",
    response_class=StreamingResponse,
    responses=_SSE_DOC,
)
async def resume_run(
    request: Request,
    run_id: Annotated[str, Path(pattern=_ID)],
    principal: Annotated[Principal, Depends(get_principal)],
    supervisor: Annotated[RunSupervisor, Depends(get_supervisor)],
    last_event_id: Annotated[int, Header(ge=0)] = 0,
) -> StreamingResponse:
    if await supervisor.owner(run_id) != principal.subject:
        raise NotFoundError("run not found")  # same answer for 'missing' and 'not yours'
    hb = request.app.state.settings.app.sse_heartbeat_seconds
    return _stream(supervisor, run_id, last_event_id, hb)
