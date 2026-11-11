"""Test-only scripted model, injected at LiteLLM's ``completion`` seam.

Everything above the seam is the real code: ``LiteLLMClient`` (payload building, stream parsing,
usage, error mapping) and the retry/breaker/fallback/metering stack. Only the network call is
replaced, with canned litellm-shaped responses.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from types import SimpleNamespace
from typing import Any

from agent_svc.adapters.llm.litellm_client import LiteLLMClient
from agent_svc.domain.exceptions import DomainError, UpstreamError
from agent_svc.domain.models import Message, ToolCall
from agent_svc.domain.ports import GoldenCase, LLMChunk, LLMResponse


def _to_messages(payload: Sequence[dict[str, Any]]) -> list[Message]:
    return [
        Message(
            m["role"],
            m.get("content") or "",
            name=m.get("name"),
            tool_call_id=m.get("tool_call_id"),
        )
        for m in payload
    ]


def _tool_call(c: ToolCall, index: int = 0) -> SimpleNamespace:
    fn = SimpleNamespace(name=c.name, arguments=json.dumps(c.arguments))
    return SimpleNamespace(id=c.id, index=index, function=fn)


class ScriptedModel:
    """An ``LLMClient`` (real ``LiteLLMClient`` over a scripted completion function)."""

    def __init__(
        self,
        name: str = "test/primary",
        *,
        reply: str = "Hello from the scripted model",
        fail_times: int = 0,
        error: DomainError | None = None,
        delay: float = 0.0,
        fail_after_chunks: int | None = None,
        tool_plan: Sequence[ToolCall] = (),
        rules: Mapping[str, tuple[str, Sequence[ToolCall]]] | None = None,
    ) -> None:
        self.name, self.reply, self.delay = name, reply, delay
        self.tool_plan = tuple(tool_plan)
        self.rules = dict(rules or {})  # last user text -> (reply, tool plan)
        self.fail_times = fail_times
        self.error = error or UpstreamError("scripted provider failure")
        self.fail_after_chunks = fail_after_chunks
        self.calls = 0
        self.seen: list[list[Message]] = []  # messages of every call, for assertions
        self._client = LiteLLMClient(name, completion=self._completion)

    # --- LLMClient port (delegates to the real adapter) ---------------------------------------
    async def complete(
        self, messages: Sequence[Message], *, model: str | None = None, **params: Any
    ) -> LLMResponse:
        return await self._client.complete(messages, model=model, **params)

    def stream(
        self, messages: Sequence[Message], *, model: str | None = None, **params: Any
    ) -> AsyncIterator[LLMChunk]:
        return self._client.stream(messages, model=model, **params)

    # --- the scripted "network" ---------------------------------------------------------------
    def _select(self, messages: Sequence[Message]) -> tuple[str, Sequence[ToolCall]]:
        if self.rules:
            last = next((m.content for m in reversed(messages) if m.role == "user"), "")
            if last in self.rules:
                return self.rules[last]
        return self.reply, self.tool_plan

    def _next_call(self, messages: Sequence[Message], tools: Any) -> ToolCall | None:
        plan = self._select(messages)[1]
        if not plan or not tools:
            return None
        done = 0
        for m in reversed(messages):
            if m.role == "user":
                break
            if m.role == "tool":
                done += 1
        return plan[done] if done < len(plan) else None

    @staticmethod
    def _usage(messages: Sequence[Message], text: str) -> SimpleNamespace:
        return SimpleNamespace(
            prompt_tokens=sum(len(m.content.split()) for m in messages),
            completion_tokens=len(text.split()),
        )

    async def _completion(self, **kw: Any) -> Any:
        messages = _to_messages(kw["messages"])
        self.calls += 1
        self.seen.append(messages)
        if self.fail_times == -1 or self.calls <= self.fail_times:
            raise self.error
        call = self._next_call(messages, kw.get("tools"))
        reply = "" if call else self._select(messages)[0]
        if kw.get("stream"):
            return self._stream(messages, call, reply)
        if self.delay:
            await asyncio.sleep(self.delay)
        msg = SimpleNamespace(content=reply, tool_calls=[_tool_call(call)] if call else None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg)], usage=self._usage(messages, reply)
        )

    async def _stream(self, messages: list[Message], call: ToolCall | None, reply: str) -> Any:
        def part(content: str = "", calls: Any = None, usage: Any = None) -> SimpleNamespace:
            delta = SimpleNamespace(content=content, tool_calls=calls)
            return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=usage)

        if call:
            yield part(calls=[_tool_call(call)])
        else:
            words = reply.split()
            for i, w in enumerate(words):
                if self.fail_after_chunks is not None and i >= self.fail_after_chunks:
                    raise self.error
                if self.delay:
                    await asyncio.sleep(self.delay)
                yield part(w + (" " if i < len(words) - 1 else ""))
        yield SimpleNamespace(choices=[], usage=self._usage(messages, reply))


def simulation_rules(cases: Sequence[GoldenCase]) -> dict[str, tuple[str, tuple[ToolCall, ...]]]:
    """Scripted replies for golden cases that carry ``metadata.simulate``."""
    rules: dict[str, tuple[str, tuple[ToolCall, ...]]] = {}
    for case in cases:
        sim = case.metadata.get("simulate")
        if not sim:
            continue
        calls = tuple(
            ToolCall(f"sim-{case.id}-{i}", t["name"], t.get("arguments", {}))
            for i, t in enumerate(sim.get("tools", []))
        )
        rules[case.input] = (sim.get("reply", ""), calls)
    return rules
