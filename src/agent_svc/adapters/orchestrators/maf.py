"""Microsoft Agent Framework adapter.

* ``PortChatClient`` - an MAF chat client backed by our resilient ``LLMClient`` port (retry,
  circuit breaker, fallback, metering). MAF's ``FunctionInvocationLayer`` runs the tool loop.
* ``MafRunner``      - maps ``Agent.run(stream=True)`` updates to framework-neutral events.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from agent_framework import (
    Agent,
    BaseChatClient,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    FunctionInvocationConfiguration,
    FunctionInvocationLayer,
    FunctionTool,
    ResponseStream,
)
from agent_framework import (
    Message as MafMessage,
)

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
    if isinstance(raw, Mapping):
        return dict(raw)
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _to_domain(messages: Sequence[MafMessage]) -> list[Message]:
    out: list[Message] = []
    for m in messages:
        role = _ROLES.get(str(m.role), "user")
        text: list[str] = []
        calls: list[ToolCall] = []
        for c in m.contents:
            if c.type == "function_result":
                body = str(c.result) if c.exception is None else f"Error: {c.exception}"
                out.append(Message("tool", body, name=c.name, tool_call_id=c.call_id))
            elif c.type == "function_call":
                calls.append(ToolCall(str(c.call_id), str(c.name), _args(c.arguments)))
            elif c.type == "text" and c.text:
                text.append(c.text)
        if text or calls:
            out.append(Message(role, "".join(text), tool_calls=tuple(calls)))
    return out


def _openai_tools(options: Mapping[str, Any]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for t in options.get("tools") or []:
        if isinstance(t, FunctionTool):
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or t.name,
                        "parameters": t.parameters(),
                    },
                }
            )
    return tools


def _usage_content(u: Usage) -> Content:
    return Content.from_usage(
        {
            "input_token_count": u.input_tokens,
            "output_token_count": u.output_tokens,
            "total_token_count": u.total_tokens,
        }
    )


def _call_contents(calls: Sequence[ToolCall]) -> list[Content]:
    return [
        Content.from_function_call(c.id, c.name, arguments=json.dumps(c.arguments)) for c in calls
    ]


class PortChatClient(FunctionInvocationLayer, BaseChatClient):
    """MAF chat client over the LLMClient port. Cost rides in ``additional_properties``."""

    def __init__(self, llm: LLMClient, *, max_iterations: int = 8) -> None:
        super().__init__(
            function_invocation_configuration=FunctionInvocationConfiguration(
                max_iterations=max_iterations, include_detailed_errors=True
            )
        )
        self._llm = llm

    def _inner_get_response(
        self,
        *,
        messages: Sequence[MafMessage],
        stream: bool,
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> Any:
        msgs = _to_domain(messages)
        tools = _openai_tools(options)
        params: dict[str, Any] = {"tools": tools} if tools else {}

        if not stream:

            async def once() -> ChatResponse:
                r = await self._llm.complete(msgs, **params)
                contents = ([Content.from_text(r.text)] if r.text else []) + _call_contents(
                    r.tool_calls
                )
                return ChatResponse(
                    messages=[MafMessage("assistant", contents)],
                    model=r.model,
                    usage_details={
                        "input_token_count": r.usage.input_tokens,
                        "output_token_count": r.usage.output_tokens,
                        "total_token_count": r.usage.total_tokens,
                    },
                    additional_properties={"cost_usd": r.usage.cost_usd},
                )

            return once()

        async def updates() -> AsyncIterator[ChatResponseUpdate]:
            async for chunk in self._llm.stream(msgs, **params):
                contents: list[Content] = []
                if chunk.delta:
                    contents.append(Content.from_text(chunk.delta))
                contents.extend(_call_contents(chunk.tool_calls))
                if chunk.usage:
                    contents.append(_usage_content(chunk.usage))
                if contents:
                    yield ChatResponseUpdate(
                        contents=contents,
                        role="assistant",
                        model=chunk.model or None,
                        additional_properties=(
                            {"cost_usd": chunk.usage.cost_usd} if chunk.usage else None
                        ),
                    )

        return ResponseStream(updates(), finalizer=ChatResponse.from_updates)


class MafRunner:
    def __init__(self, settings: Settings, llm: LLMClient, tools: ToolProvider) -> None:
        self._settings, self._llm, self._tools = settings, llm, tools
        # MAF stops its own loop silently (it injects a sentinel answer), so give it one spare
        # iteration and enforce the real cap in stream(), where we can raise IterationLimitError.
        self._client = PortChatClient(llm, max_iterations=settings.orchestrator.max_iterations + 1)
        self._agent: Agent | None = None
        self._specs: dict[str, ToolSpec] = {}

    async def _ensure(self) -> Agent:
        if self._agent is None:
            specs = await self._tools.list_tools()
            self._specs = {s.name: s for s in specs}
            self._agent = Agent(
                self._client,
                self._settings.orchestrator.system_prompt,
                name="agent",
                tools=[self._wrap(s) for s in specs],
            )
        return self._agent

    def _wrap(self, spec: ToolSpec) -> FunctionTool:
        provider = self._tools

        async def run(**kwargs: Any) -> str:
            try:
                return await provider.call(spec.name, kwargs)
            except ToolExecutionError as exc:
                raise RuntimeError(
                    exc.detail
                ) from exc  # MAF reports it as a failed function result

        return FunctionTool(
            name=spec.name,
            description=spec.description or spec.name,
            func=run,
            input_model=spec.input_schema or {"type": "object", "properties": {}},
        )

    async def stream(self, request: AgentRequest) -> AsyncIterator[AgentEvent]:
        rid = request.run_id
        agent = await self._ensure()
        yield RunStarted(
            run_id=rid,
            thread_id=request.thread_id,
            agent_name=request.agent_name,
            agent_version=request.agent_version,
        )
        system = "\n".join(m.content for m in request.messages if m.role == "system")
        convo = [
            MafMessage("user" if m.role == "user" else "assistant", [Content.from_text(m.content)])
            for m in request.messages
            if m.role != "system"
        ]
        if system:
            convo.insert(0, MafMessage("system", [Content.from_text(system)]))

        started: dict[str, float] = {}
        text_out: list[str] = []
        total = Usage()
        awaiting_answer = False  # a tool result arrived and no text has followed yet
        tool_rounds = 0
        limit = self._settings.orchestrator.max_iterations
        async for upd in agent.run(convo, stream=True):
            for c in upd.contents:
                if c.type == "text" and c.text:
                    awaiting_answer = False
                    text_out.append(c.text)
                    yield MessageDelta(run_id=rid, text=c.text)
                elif c.type == "usage" and c.usage_details:
                    d = c.usage_details
                    cost = float((upd.additional_properties or {}).get("cost_usd", 0.0))
                    usage = Usage(
                        d.get("input_token_count") or 0, d.get("output_token_count") or 0, cost
                    )
                    total = total + usage
                    yield UsageReported(
                        run_id=rid,
                        usage=usage,
                        model=str(getattr(upd.raw_representation, "model", None) or ""),
                    )
                elif c.type == "function_call" and c.call_id and c.call_id not in started:
                    tool_rounds += 1
                    if tool_rounds > limit:
                        raise IterationLimitError(f"agent exceeded {limit} tool-call iterations")
                    started[c.call_id] = time.perf_counter()
                    yield StepStarted(
                        run_id=rid,
                        step_id=c.call_id,
                        kind=step_kind_for(self._specs.get(str(c.name))),
                        name=str(c.name),
                        input={"input": summarize(_args(c.arguments), 300)},
                    )
                elif c.type == "function_result" and c.call_id in started:
                    awaiting_answer = True
                    yield StepFinished(
                        run_id=rid,
                        step_id=str(c.call_id),
                        ok=c.exception is None,
                        output_summary=summarize(
                            c.result if c.exception is None else c.exception, 300
                        ),
                        duration_ms=int((time.perf_counter() - started.pop(str(c.call_id))) * 1000),
                    )
        if awaiting_answer:  # MAF stops its loop at max_iterations without raising
            raise IterationLimitError(f"agent exceeded {limit} iterations without finishing")
        yield MessageCompleted(run_id=rid, text="".join(text_out))
        yield RunCompleted(run_id=rid, usage=total)


def build(settings: Settings, llm: LLMClient, tools: ToolProvider) -> MafRunner:
    return MafRunner(settings, llm, tools)
