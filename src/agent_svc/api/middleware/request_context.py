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

    def __init__(self, app: ASGIApp, *, agent_version: str = "") -> None:
        self.app = app
        self.agent_version = agent_version

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
