from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Request
from fastapi.responses import StreamingResponse
from pydantic import TypeAdapter

from agent_svc.api.deps import get_chat_service, get_principal
from agent_svc.api.schemas.chat import ChatRequest, SseEvent
from agent_svc.api.sse import sse_stream
from agent_svc.application.chat_service import ChatService
from agent_svc.domain.exceptions import BadRequestError, PayloadTooLargeError
from agent_svc.domain.models import Message, Principal

router = APIRouter(tags=["chat"])
_THREAD_ID = r"^[A-Za-z0-9_-]{1,64}$"


@router.post(
    "/threads/{thread_id}/chat",
    summary="Chat with the agent (Server-Sent Events)",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {"schema": TypeAdapter(SseEvent).json_schema()}},
            "description": "Stream of typed events: run.started, step.started/finished "
            "(tool/rag/agent/mcp/guardrail, nested via parent_id), message.delta, usage, "
            "message.completed, run.error, run.completed.",
        }
    },
)
async def chat(
    request: Request,
    body: ChatRequest,
    thread_id: Annotated[str, Path(pattern=_THREAD_ID)],
    principal: Annotated[Principal, Depends(get_principal)],
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> StreamingResponse:
    cfg = request.app.state.settings.app
    if len(body.messages) > cfg.max_messages_per_request:
        raise BadRequestError(f"at most {cfg.max_messages_per_request} messages per request")
    if any(len(m.content) > cfg.max_message_chars for m in body.messages):
        raise PayloadTooLargeError(f"message exceeds {cfg.max_message_chars} characters")
    if body.messages[-1].role != "user":
        raise BadRequestError("the last message must have role 'user'")

    areq = service.new_request(
        thread_id, principal, [Message(m.role, m.content) for m in body.messages]
    )
    stream = sse_stream(service.stream(areq), heartbeat_seconds=cfg.sse_heartbeat_seconds)
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Run-ID": areq.run_id},
    )
