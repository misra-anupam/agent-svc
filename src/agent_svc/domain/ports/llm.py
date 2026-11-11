from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent_svc.domain.models import Message, Usage


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    model: str
    usage: Usage = Usage()
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LLMChunk:
    delta: str = ""
    usage: Usage | None = None  # set on the final chunk when the provider reports it
    model: str = ""


class LLMClient(Protocol):
    async def complete(
        self, messages: Sequence[Message], *, model: str | None = None, **params: Any
    ) -> LLMResponse: ...

    def stream(
        self, messages: Sequence[Message], *, model: str | None = None, **params: Any
    ) -> AsyncIterator[LLMChunk]: ...
