from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, Request

from agent_svc.api.sse import RunSupervisor
from agent_svc.application.chat_service import ChatService
from agent_svc.application.memory_facade import MemoryFacade
from agent_svc.domain.exceptions import ForbiddenError, UnavailableError
from agent_svc.domain.models import Principal
from agent_svc.domain.ports import Cache


def client_ip(request: Request) -> str | None:
    hops: int = request.app.state.settings.app.trusted_proxy_hops
    if hops > 0:
        xff = [
            p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()
        ]
        if len(xff) >= hops:
            return xff[-hops]  # nth address from the right = what our outermost proxy saw
    return request.client.host if request.client else None


async def get_principal(
    request: Request, authorization: Annotated[str | None, Header()] = None
) -> Principal:
    container = request.app.state.container
    if container.authenticator is None:
        raise UnavailableError("service starting")
    principal: Principal = await container.authenticator.authenticate(
        authorization, client_ip=client_ip(request)
    )
    return principal


def require_scope(scope: str):  # type: ignore[no-untyped-def]
    async def dep(principal: Annotated[Principal, Depends(get_principal)]) -> Principal:
        if (
            not principal.anonymous
            and scope not in principal.scopes
            and "*" not in principal.scopes
        ):
            raise ForbiddenError(f"missing scope {scope!r}")
        return principal

    return dep


def get_chat_service(request: Request) -> ChatService:
    c = request.app.state.container
    if c.runner is None:
        raise UnavailableError("agent runner not ready")
    s = request.app.state.settings.app
    return ChatService(
        c.runner,
        agent_name=s.agent_name,
        agent_version=s.agent_version,
        memory=c.memory if c.memory.stateful else None,
        admission=c.admission,
    )


def get_supervisor(request: Request) -> RunSupervisor:
    sup: RunSupervisor = request.app.state.supervisor
    return sup


def get_memory(request: Request) -> MemoryFacade:
    mem: MemoryFacade = request.app.state.container.memory
    return mem


def get_kv(request: Request) -> Cache | None:
    """Redis-backed key/value store; ``None`` when Redis is off (no idempotency)."""
    kv: Cache | None = request.app.state.container.kv
    return kv
