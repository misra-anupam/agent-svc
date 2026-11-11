from __future__ import annotations

from fastapi import Request

from agent_svc.application.chat_service import ChatService
from agent_svc.domain.exceptions import UnavailableError
from agent_svc.domain.models import Principal


def get_principal(request: Request) -> Principal:
    """Anonymous principal keyed by client IP. Replaced by JWT auth in phase 3."""
    ip = request.client.host if request.client else None
    return Principal(subject=f"ip:{ip}", ip=ip, anonymous=True)


def get_chat_service(request: Request) -> ChatService:
    container = request.app.state.container
    if container.runner is None:
        raise UnavailableError("agent runner not ready")
    s = request.app.state.settings.app
    return ChatService(container.runner, agent_name=s.agent_name, agent_version=s.agent_version)
