from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class Resource(Protocol):
    """Anything with managed startup/shutdown (clients, pools, sidecars)."""

    name: str

    async def start(self) -> None: ...
    async def stop(self) -> None: ...


@runtime_checkable
class HealthCheck(Protocol):
    name: str
    required: bool

    async def ping(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    ready: bool
    checks: dict[str, bool]


class ReadinessProvider(Protocol):
    async def readiness(self) -> ReadinessReport: ...
