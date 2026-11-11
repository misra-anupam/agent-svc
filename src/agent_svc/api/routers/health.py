from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agent_svc.domain.ports import ReadinessProvider

router = APIRouter(tags=["ops"])


class Liveness(BaseModel):
    status: str = "ok"


class Readiness(BaseModel):
    status: str
    checks: dict[str, bool]


@router.get("/healthz", response_model=Liveness, summary="Liveness probe")
async def healthz() -> Liveness:
    return Liveness()


@router.get("/readyz", response_model=Readiness, summary="Readiness probe")
async def readyz(request: Request) -> JSONResponse:
    provider: ReadinessProvider = request.app.state.container
    report = await provider.readiness()
    body = Readiness(status="ready" if report.ready else "unavailable", checks=report.checks)
    return JSONResponse(body.model_dump(), status_code=200 if report.ready else 503)
