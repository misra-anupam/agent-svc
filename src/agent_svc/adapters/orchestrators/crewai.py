"""CrewAI adapter.

CrewAI runs its agent loop synchronously in a worker thread. We keep it behind our ports:

* ``PortCrewLLM``  - a ``BaseLLM`` that forwards to the resilient ``LLMClient`` port (retry,
  breaker, fallback, metering) and returns native tool-call lists, so CrewAI's executor runs the
  tool loop. It hops back onto the service's event loop for every call.
* ``ProviderCrewTool`` - a ``BaseTool`` that forwards to the ``ToolProvider`` (allowlist, schema
  validation, timeout) and emits step events.
* ``CrewRunner``   - runs ``Agent.kickoff_async`` and drains an event queue fed by the two above.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable
from concurrent.futures import Future
from typing import Any

from crewai import Agent
from crewai.llms.base_llm import BaseLLM
from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, PrivateAttr, create_model

from agent_svc.adapters.orchestrators._text import summarize
from agent_svc.config.settings import Settings
from agent_svc.domain.exceptions import IterationLimitError, ToolExecutionError
from agent_svc.domain.models import (
    AgentEvent,
    AgentRequest,
    Message,
    MessageCompleted,
    MessageDelta,
    Role,
    RunCompleted,
    RunStarted,
    StepFinished,
    StepStarted,
    ToolCall,
    ToolSpec,
    Usage,
    UsageReported,
)
from agent_svc.domain.ports import LLMClient, ToolProvider, step_kind_for

_ROLES: dict[str, Role] = {
    "user": "user",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
}


def _args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return "" if content is None else str(content)


def _to_domain(messages: str | list[Any]) -> list[Message]:
    if isinstance(messages, str):
        return [Message("user", messages)]
    out: list[Message] = []
    for m in messages:
        role = _ROLES.get(str(m.get("role")), "user")
        calls = tuple(
            ToolCall(str(c.get("id")), c["function"]["name"], _args(c["function"].get("arguments")))
            for c in (m.get("tool_calls") or [])
        )
        out.append(
            Message(
                role,
                _text(m.get("content")),
                name=m.get("name"),
                tool_calls=calls,
                tool_call_id=m.get("tool_call_id"),
            )
        )
    return out


class _Bridge:
    """Thread-safe hand-off between CrewAI's worker thread and the service's event loop."""

    def __init__(self, loop: asyncio.AbstractEventLoop, max_rounds: int) -> None:
        self.loop = loop
        self.queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
        self.max_rounds = max_rounds
        self.rounds = 0
        self.limit_hit = False
        self.usage = Usage()
        self.model = ""
        self.text: list[str] = []
        self.tool_failed: dict[str, bool] = {}

    def emit(self, event: AgentEvent) -> None:
        self.loop.call_soon_threadsafe(self.queue.put_nowait, event)

    def run(self, coro: Any) -> Any:
        fut: Future[Any] = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return fut.result()


class PortCrewLLM(BaseLLM):
    """CrewAI ``BaseLLM`` over the LLMClient port."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    _llm: Any = PrivateAttr(default=None)
    _bridge: Any = PrivateAttr(default=None)

    def attach(self, llm: LLMClient, bridge: _Bridge) -> None:
        self._llm, self._bridge = llm, bridge

    def supports_function_calling(self) -> bool:
        return True

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 128_000

    async def _complete(
        self, msgs: list[Message], tools: list[dict[str, Any]] | None
    ) -> tuple[str, list[ToolCall]]:
        b: _Bridge = self._bridge
        params: dict[str, Any] = {"tools": tools} if tools else {}
        calls: list[ToolCall] = []
        parts: list[str] = []
        async for chunk in self._llm.stream(msgs, **params):
            if chunk.delta:
                parts.append(chunk.delta)
                b.emit(MessageDelta(run_id="", text=chunk.delta))
            calls.extend(chunk.tool_calls)
            if chunk.model:
                b.model = chunk.model
            if chunk.usage:
                b.usage = b.usage + chunk.usage
                b.emit(UsageReported(run_id="", usage=chunk.usage, model=chunk.model or ""))
        return "".join(parts), calls

    def call(
        self,
        messages: str | list[Any],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | list[dict[str, Any]]:
        b: _Bridge = self._bridge
        text, calls = b.run(self._complete(_to_domain(messages), tools))
        if not calls:
            return str(text)
        b.rounds += 1
        if b.rounds > b.max_rounds:
            b.limit_hit = True
            raise IterationLimitError(f"agent exceeded {b.max_rounds} tool-call rounds")
        return [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
            }
            for c in calls
        ]

    async def acall(  # type: ignore[override]
        self, messages: str | list[Any], tools: Any = None, **kwargs: Any
    ) -> str | list[dict[str, Any]]:
        return await asyncio.to_thread(self.call, messages, tools, **kwargs)


def _schema_model(spec: ToolSpec) -> type[BaseModel]:
    """A permissive pydantic model whose JSON schema *is* the tool's JSON schema."""
    schema = dict(spec.input_schema or {"type": "object", "properties": {}})

    def replace(target: dict[str, Any]) -> None:
        target.clear()
        target.update(schema)

    return create_model(
        f"{spec.name}_args",
        __config__=ConfigDict(extra="allow", json_schema_extra=replace),
    )


class ProviderCrewTool(BaseTool):
    """A ToolProvider tool exposed to CrewAI. Step events are emitted from here."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    _provider: Any = PrivateAttr(default=None)
    _bridge: Any = PrivateAttr(default=None)
    _spec: Any = PrivateAttr(default=None)

    def attach(self, provider: ToolProvider, bridge: _Bridge, spec: ToolSpec) -> None:
        self._provider, self._bridge, self._spec = provider, bridge, spec

    def _run(self, **kwargs: Any) -> str:
        b: _Bridge = self._bridge
        spec: ToolSpec = self._spec
        step_id = f"{spec.name}:{time.monotonic_ns()}"
        kind = step_kind_for(spec)
        b.emit(
            StepStarted(
                run_id="",
                step_id=step_id,
                kind=kind,
                name=spec.name,
                input={"input": summarize(kwargs, 300)},
            )
        )
        started = time.monotonic()
        try:
            result: str = b.run(self._provider.call(spec.name, kwargs))
            ok = True
        except ToolExecutionError as exc:
            result, ok = f"Error: {exc.detail}", False
        b.emit(
            StepFinished(
                run_id="",
                step_id=step_id,
                ok=ok,
                output_summary=summarize(result, 300),
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        )
        return result


def _make_tool(
    spec: ToolSpec, provider: ToolProvider, bridge: _Bridge, factory: Callable[..., Any]
) -> ProviderCrewTool:
    tool = factory(
        name=spec.name,
        description=spec.description or spec.name,
        args_schema=_schema_model(spec),
    )
    tool.attach(provider, bridge, spec)
    return tool  # type: ignore[no-any-return]


class CrewRunner:
    def __init__(self, settings: Settings, llm: LLMClient, tools: ToolProvider) -> None:
        self._settings, self._llm, self._tools = settings, llm, tools
        self._specs: list[ToolSpec] | None = None

    async def stream(self, request: AgentRequest) -> AsyncIterator[AgentEvent]:
        rid = request.run_id
        if self._specs is None:
            self._specs = await self._tools.list_tools()
        limit = self._settings.orchestrator.max_iterations
        bridge = _Bridge(asyncio.get_running_loop(), limit)
        llm = PortCrewLLM(model="port")
        llm.attach(self._llm, bridge)
        tools = [_make_tool(s, self._tools, bridge, ProviderCrewTool) for s in self._specs]
        agent = Agent(
            role="Assistant",
            goal="Answer the user's request accurately, using tools when they help.",
            backstory=self._settings.orchestrator.system_prompt,
            llm=llm,
            tools=tools,
            max_iter=limit + 5,  # our bridge enforces the real cap and raises
            allow_delegation=False,
            verbose=False,
        )
        yield RunStarted(
            run_id=rid,
            thread_id=request.thread_id,
            agent_name=request.agent_name,
            agent_version=request.agent_version,
        )
        convo: list[Any] = [
            {"role": m.role if m.role in ("user", "assistant") else "user", "content": m.content}
            for m in request.messages
            if m.role != "system"
        ]
        task = asyncio.create_task(agent.kickoff_async(convo))
        try:
            while not (task.done() and bridge.queue.empty()):
                getter = asyncio.create_task(bridge.queue.get())
                await asyncio.wait({getter, task}, return_when=asyncio.FIRST_COMPLETED)
                if getter.done():
                    ev = getter.result()
                    ev = _with_run(ev, rid)
                    if isinstance(ev, MessageDelta):
                        bridge.text.append(ev.text)
                    yield ev
                else:
                    getter.cancel()
            exc = task.exception()
            if exc is not None:
                if bridge.limit_hit:
                    raise IterationLimitError(f"agent exceeded {limit} tool-call rounds") from exc
                raise exc
            final = getattr(task.result(), "raw", "") or "".join(bridge.text)
        finally:
            if not task.done():
                task.cancel()
        yield MessageCompleted(run_id=rid, text=str(final))
        yield RunCompleted(run_id=rid, usage=bridge.usage)


def _with_run(ev: AgentEvent, rid: str) -> AgentEvent:
    from dataclasses import replace

    return replace(ev, run_id=rid)


def build(settings: Settings, llm: LLMClient, tools: ToolProvider) -> CrewRunner:
    return CrewRunner(settings, llm, tools)
