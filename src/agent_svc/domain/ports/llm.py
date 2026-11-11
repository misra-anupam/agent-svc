from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent_svc.domain.models import Message, ToolCall, Usage


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    model: str
    usage: Usage = Usage()
    tool_calls: tuple[ToolCall, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LLMChunk:
    delta: str = ""
    usage: Usage | None = None  # set on the final chunk when the provider reports it
    tool_calls: tuple[ToolCall, ...] = ()  # complete calls, emitted once the stream has them
    model: str = ""


class LLMClient(Protocol):
    async def complete(
        self, messages: Sequence[Message], *, model: str | None = None, **params: Any
    ) -> LLMResponse: ...

    def stream(
        self, messages: Sequence[Message], *, model: str | None = None, **params: Any
    ) -> AsyncIterator[LLMChunk]: ...
