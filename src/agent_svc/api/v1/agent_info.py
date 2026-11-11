from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(tags=["agent"])


class AgentInfo(BaseModel):
    name: str
    version: str
    api_version: str = "v1"


@router.get("/agent", response_model=AgentInfo, summary="Agent name and version")
async def agent_info(request: Request) -> AgentInfo:
    s = request.app.state.settings.app
    return AgentInfo(name=s.agent_name, version=s.agent_version)
