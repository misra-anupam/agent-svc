"""RFC 9457 problem+json rendering for every error path."""

from __future__ import annotations

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from agent_svc.domain.exceptions import DomainError, RateLimitedError

log = structlog.get_logger()
PROBLEM = "application/problem+json"


class Problem(BaseModel):
    type: str = "about:blank"
    title: str
    status: int
    detail: str | None = None
    code: str
    instance: str | None = None
    request_id: str | None = None
    errors: list[dict[str, object]] | None = None


def _problem(
    request: Request,
    status: int,
    title: str,
    code: str,
    detail: str | None,
    errors: list[dict[str, object]] | None = None,
) -> JSONResponse:
    body = Problem(
        title=title,
        status=status,
        detail=detail,
        code=code,
        instance=request.url.path,
        request_id=getattr(request.state, "request_id", None),
        errors=errors,
    )
    return JSONResponse(body.model_dump(exclude_none=True), status_code=status, media_type=PROBLEM)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(DomainError)
    async def _domain(request: Request, exc: DomainError) -> JSONResponse:
        if exc.status_code >= 500:
            log.error("domain_error", code=exc.code, detail=exc.detail)
        resp = _problem(request, exc.status_code, exc.title, exc.code, exc.detail)
        if isinstance(exc, RateLimitedError):
            resp.headers["Retry-After"] = str(exc.retry_after)
        return resp

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return _problem(
            request,
            422,
            "Validation Error",
            "validation_error",
            "Request body failed validation",
            errors=errors,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _problem(request, exc.status_code, str(exc.detail), f"http_{exc.status_code}", None)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled_error")
        return _problem(request, 500, "Internal Server Error", "internal_error", None)
