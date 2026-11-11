from __future__ import annotations

from collections.abc import AsyncIterator

import fakeredis
import httpx
import pytest

from agent_svc.adapters.infra.redis import RedisResource
from agent_svc.config import Settings
from agent_svc.container import Container
from agent_svc.main import create_app


@pytest.fixture
def settings() -> Settings:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    s.app.environment = "test"
    s.app.log_json = False
    s.redis.enabled = False  # tests inject a fake resource instead
    return s


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    fake = RedisResource(settings.redis, client=fakeredis.FakeAsyncRedis())
    fake.required = True
    app = create_app(settings, Container(settings, extra_resources=[fake]))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
