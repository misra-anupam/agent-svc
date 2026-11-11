"""Deterministic LLM for tests/demos with scripted failures."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any

from agent_svc.config.settings import Settings
from agent_svc.domain.exceptions import DomainError, UpstreamError
from agent_svc.domain.models import Message, ToolCall, Usage
from agent_svc.domain.ports import LLMChunk, LLMResponse


class FakeLLM:
    def __init__(
        self,
        name: str = "fake/primary",
        *,
        reply: str = "Hello from the fake model",
        fail_times: int = 0,
        error: DomainError | None = None,
        delay: float = 0.0,
        fail_after_chunks: int | None = None,
        tool_plan: Sequence[ToolCall] = (),
    ) -> None:
        self.tool_plan = tuple(tool_plan)
        self.name, self.reply, self.delay = name, reply, delay
        self.fail_times = fail_times
        self.error = error or UpstreamError("fake provider failure")
        self.fail_after_chunks = fail_after_chunks
        self.calls = 0
        self.seen: list[list[Message]] = []  # messages of every call, for assertions

    def _maybe_fail(self, messages: Sequence[Message] = ()) -> None:
        self.calls += 1
        self.seen.append(list(messages))
        if self.fail_times == -1 or self.calls <= self.fail_times:
            raise self.error

    def _next_tool_call(
        self, messages: Sequence[Message], params: dict[str, Any]
    ) -> ToolCall | None:
        """Stateless script: plan[i] is issued until i+1 tool results follow the last user turn."""
        if not self.tool_plan or not params.get("tools"):
            return None
        done = 0
        for m in reversed(messages):
            if m.role == "user":
                break
            if m.role == "tool":
                done += 1
        return self.tool_plan[done] if done < len(self.tool_plan) else None

    @staticmethod
    def _usage(messages: Sequence[Message], text: str) -> Usage:
        return Usage(sum(len(m.content.split()) for m in messages), len(text.split()))

    async def complete(
        self, messages: Sequence[Message], *, model: str | None = None, **params: Any
    ) -> LLMResponse:
        self._maybe_fail(messages)
        if self.delay:
            await asyncio.sleep(self.delay)
        call = self._next_tool_call(messages, params)
        if call:
            return LLMResponse("", self.name, self._usage(messages, ""), tool_calls=(call,))
        return LLMResponse(self.reply, self.name, self._usage(messages, self.reply))

    async def stream(
        self, messages: Sequence[Message], *, model: str | None = None, **params: Any
    ) -> AsyncIterator[LLMChunk]:
        self._maybe_fail(messages)
        call = self._next_tool_call(messages, params)
        if call:
            yield LLMChunk(tool_calls=(call,), model=self.name)
            yield LLMChunk(usage=self._usage(messages, ""), model=self.name)
            return
        words = self.reply.split()
        for i, w in enumerate(words):
            if self.fail_after_chunks is not None and i >= self.fail_after_chunks:
                raise self.error
            if self.delay:
                await asyncio.sleep(self.delay)
            yield LLMChunk(delta=w + (" " if i < len(words) - 1 else ""), model=self.name)
        yield LLMChunk(usage=self._usage(messages, self.reply), model=self.name)


def build_fake(_settings: Settings, model: str) -> FakeLLM:
    return FakeLLM(model)
