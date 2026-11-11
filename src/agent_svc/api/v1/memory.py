from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Path, Response
from pydantic import BaseModel, ConfigDict, Field

from agent_svc.api.deps import get_memory, get_principal
from agent_svc.application.memory_facade import MemoryFacade
from agent_svc.domain.exceptions import NotFoundError
from agent_svc.domain.models import Principal

router = APIRouter(tags=["memory"])


class MessageOut(BaseModel):
    role: str
    content: str


class ThreadOut(BaseModel):
    thread_id: str
    messages: list[MessageOut]


class UserFactsIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    facts: dict[str, Any] = Field(min_length=1, max_length=50)


class UserFactsOut(BaseModel):
    facts: dict[str, Any]


@router.get("/threads/{thread_id}", response_model=ThreadOut, summary="Read a thread's history")
async def get_thread(
    thread_id: Annotated[str, Path(pattern=r"^[A-Za-z0-9_-]{1,64}$")],
    principal: Annotated[Principal, Depends(get_principal)],
    memory: Annotated[MemoryFacade, Depends(get_memory)],
) -> ThreadOut:
    if memory.thread is None:
        raise NotFoundError("server-side thread memory is disabled")
    msgs = await memory.thread.load(principal.tenant, memory.thread_key(principal, thread_id))
    if not msgs:
        raise NotFoundError("thread not found")
    return ThreadOut(
        thread_id=thread_id, messages=[MessageOut(role=m.role, content=m.content) for m in msgs]
    )


@router.delete("/threads/{thread_id}", status_code=204, summary="Delete a thread")
async def delete_thread(
    thread_id: Annotated[str, Path(pattern=r"^[A-Za-z0-9_-]{1,64}$")],
    principal: Annotated[Principal, Depends(get_principal)],
    memory: Annotated[MemoryFacade, Depends(get_memory)],
) -> Response:
    if memory.thread is not None:
        await memory.thread.delete(principal.tenant, memory.thread_key(principal, thread_id))
    return Response(status_code=204)


@router.get("/memory/user", response_model=UserFactsOut, summary="Read my durable user memory")
async def get_user_memory(
    principal: Annotated[Principal, Depends(get_principal)],
    memory: Annotated[MemoryFacade, Depends(get_memory)],
) -> UserFactsOut:
    if memory.user is None:
        raise NotFoundError("user memory is disabled")
    return UserFactsOut(facts=await memory.user.get(principal.tenant, principal.subject))


@router.put("/memory/user", response_model=UserFactsOut, summary="Merge facts into user memory")
async def put_user_memory(
    body: UserFactsIn,
    principal: Annotated[Principal, Depends(get_principal)],
    memory: Annotated[MemoryFacade, Depends(get_memory)],
) -> UserFactsOut:
    if memory.user is None:
        raise NotFoundError("user memory is disabled")
    await memory.user.merge(principal.tenant, principal.subject, body.facts)
    return UserFactsOut(facts=await memory.user.get(principal.tenant, principal.subject))


@router.delete("/memory", status_code=204, summary="Erase all of my user and long-term memory")
async def erase_my_memory(
    principal: Annotated[Principal, Depends(get_principal)],
    memory: Annotated[MemoryFacade, Depends(get_memory)],
) -> Response:
    if memory.user is not None:
        await memory.user.delete(principal.tenant, principal.subject)
    if memory.long_term is not None:
        await memory.long_term.forget(principal.tenant, principal.subject)
    return Response(status_code=204)
