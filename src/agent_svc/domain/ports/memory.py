from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent_svc.domain.models import Message


class ThreadMemory(Protocol):
    """Conversation history scoped to (tenant, thread)."""

    async def append(self, tenant: str, thread_id: str, messages: Sequence[Message]) -> None: ...
    async def load(
        self, tenant: str, thread_id: str, limit: int | None = None
    ) -> list[Message]: ...
    async def delete(self, tenant: str, thread_id: str) -> None: ...


class UserMemory(Protocol):
    """Durable per-user profile/preferences."""

    async def get(self, tenant: str, user_id: str) -> dict[str, Any]: ...
    async def merge(self, tenant: str, user_id: str, facts: dict[str, Any]) -> None: ...
    async def delete(self, tenant: str, user_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class MemoryHit:
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class LongTermMemory(Protocol):
    """Semantic long-term store, namespaced per tenant/user."""

    async def remember(self, tenant: str, user_id: str, text: str, **meta: Any) -> None: ...
    async def recall(
        self, tenant: str, user_id: str, query: str, k: int = 5
    ) -> list[MemoryHit]: ...
    async def forget(self, tenant: str, user_id: str) -> None: ...
