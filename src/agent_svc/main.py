from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from prometheus_client import make_asgi_app

from agent_svc import __version__
from agent_svc.api.errors import install_error_handlers
from agent_svc.api.middleware.request_context import RequestContextMiddleware
from agent_svc.api.routers import health
from agent_svc.api.v1 import router as v1_router
from agent_svc.config import Settings, get_settings
from agent_svc.container import Container
from agent_svc.observability.logging import configure_logging


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.app.log_level, json_logs=settings.app.log_json)
    container = container or Container(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await container.start()
        app.state.container = container
        try:
            yield
        finally:
            structlog.get_logger().info("shutting_down")
            await container.stop()

    app = FastAPI(
        title=settings.app.name,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if settings.app.environment != "production" else None,
        redoc_url=None,
    )
    app.state.settings = settings
    app.state.container = container  # available to routers even before lifespan in tests

    if settings.app.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.app.cors_origins,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type", "Idempotency-Key", "Last-Event-ID"],
        )
    app.add_middleware(RequestContextMiddleware, agent_version=settings.app.agent_version)
    install_error_handlers(app)

    app.include_router(health.router)
    app.include_router(v1_router)
    app.mount("/metrics", make_asgi_app())

    if settings.app.environment != "production":
        ui_file = Path(__file__).parent / "ui" / "index.html"

        @app.get("/ui", include_in_schema=False)
        async def demo_ui() -> FileResponse:
            return FileResponse(ui_file, headers={"Cache-Control": "no-store"})

    return app
