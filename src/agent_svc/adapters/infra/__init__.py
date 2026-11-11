from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Infra:
    """Shared clients handed to adapter factories. Fields are None when a resource is disabled."""

    redis: Any = None  # redis.asyncio.Redis
    pg_pool: Any = None  # asyncpg.Pool
