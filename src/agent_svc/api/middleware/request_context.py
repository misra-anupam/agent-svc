"""Pure-ASGI middlewares (safe for streaming/SSE responses; no body buffering)."""

from __future__ import annotations

import re
import time
import uuid

import structlog
from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from agent_svc.observability.metrics import HTTP_INFLIGHT, HTTP_LATENCY, HTTP_REQUESTS

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")


class RequestContextMiddleware:
    """Request-id propagation, access log, Prometheus HTTP metrics, security headers."""

    # Pages that legitimately load scripts/styles (Swagger UI, demo UI) get no CSP.
    _CSP_EXEMPT = ("/docs", "/ui", "/openapi.json")
    _CSP = "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"

    def __init__(self, app: ASGIApp, *, agent_version: str = "", hsts: bool = False) -> None:
        self.app = app
        self.agent_version = agent_version
        self.hsts = hsts  # only behind TLS (production): HSTS on plain HTTP is meaningless

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        req = Request(scope)
        supplied = req.headers.get("x-request-id", "")
        request_id = supplied if _SAFE_ID.match(supplied) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        start = time.perf_counter()
        status = 500
        HTTP_INFLIGHT.inc()

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = request_id
                headers["X-Agent-Version"] = self.agent_version
                headers.setdefault("X-Content-Type-Options", "nosniff")
                headers.setdefault("X-Frame-Options", "DENY")
                headers.setdefault("Referrer-Policy", "no-referrer")
                headers.setdefault("Cache-Control", "no-store")
                headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
                headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
                headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
                if not scope["path"].startswith(self._CSP_EXEMPT):
                    headers.setdefault("Content-Security-Policy", self._CSP)
                if self.hsts:
                    headers.setdefault(
                        "Strict-Transport-Security", "max-age=63072000; includeSubDomains"
                    )
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            HTTP_INFLIGHT.dec()
            route = getattr(scope.get("route"), "path", "unmatched")
            if scope["path"].startswith("/metrics"):  # mounted ASGI app: no APIRoute
                route = "/metrics"
            elapsed = time.perf_counter() - start
            HTTP_REQUESTS.labels(req.method, route, str(status)).inc()
            HTTP_LATENCY.labels(req.method, route).observe(elapsed)
            if route not in ("/healthz", "/readyz", "/metrics"):
                structlog.get_logger().info(
                    "http_request",
                    method=req.method,
                    route=route,
                    status=status,
                    ms=round(elapsed * 1000, 1),
                )
