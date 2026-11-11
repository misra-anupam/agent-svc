"""LangChain-side bridges used by the LangGraph adapter.

* ``PortChatModel``: exposes our resilient ``LLMClient`` (retry/breaker/fallback/metering) as a
  LangChain chat model with tool calling.
* ``to_langchain_tools``: exposes any ``ToolProvider`` (local tools, MCP) as LangChain tools.
* ``LangGraphRunner``: maps ``astream_events`` v2 to framework-neutral ``AgentEvent``s.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict

from agent_svc.adapters.orchestrators._text import summarize
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
    StepKind,
    StepStarted,
    ToolCall,
    ToolSpec,
    Usage,
    UsageReported,
)
from agent_svc.domain.ports import ToolProvider, step_kind_for

_ROLES: dict[str, Role] = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}


def _to_domain(msgs: list[BaseMessage]) -> list[Message]:
    out: list[Message] = []
    for m in msgs:
        role = _ROLES.get(m.type, "user")
        calls = tuple(
            ToolCall(str(c.get("id") or f"call_{i}"), c["name"], dict(c.get("args") or {}))
            for i, c in enumerate(getattr(m, "tool_calls", None) or [])
        )
        content = m.content if isinstance(m.content, str) else json.dumps(m.content)
        out.append(
            Message(
                role,
                content,
                name=getattr(m, "name", None) if role == "tool" else None,
                tool_calls=calls,
                tool_call_id=m.tool_call_id if isinstance(m, ToolMessage) else None,
            )
        )
    return out


def _usage_meta(u: Usage) -> dict[str, int]:
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "total_tokens": u.total_tokens,
    }


class PortChatModel(BaseChatModel):
    """LangChain chat model backed by the LLMClient port (async only)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm: Any

    @property
    def _llm_type(self) -> str:
        return "agent-svc-port"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise NotImplementedError("PortChatModel is async-only; use ainvoke/astream")

    @staticmethod
    def _params(kwargs: dict[str, Any]) -> dict[str, Any]:
        return {"tools": kwargs["tools"]} if kwargs.get("tools") else {}

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        resp = await self.llm.complete(_to_domain(messages), **self._params(kwargs))
        msg = AIMessage(
            content=resp.text,
            tool_calls=[
                {"id": c.id, "name": c.name, "args": c.arguments, "type": "tool_call"}
                for c in resp.tool_calls
            ],
            usage_metadata=_usage_meta(resp.usage),
            response_metadata={"model_name": resp.model, "cost_usd": resp.usage.cost_usd},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        async for chunk in self.llm.stream(_to_domain(messages), **self._params(kwargs)):
            if chunk.delta:
                if run_manager:
                    await run_manager.on_llm_new_token(chunk.delta)
                yield ChatGenerationChunk(message=AIMessageChunk(content=chunk.delta))
            if chunk.tool_calls:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            {
                                "id": c.id,
                                "name": c.name,
                                "args": json.dumps(c.arguments),
                                "index": i,
                            }
                            for i, c in enumerate(chunk.tool_calls)
                        ],
                    )
                )
            if chunk.usage:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        usage_metadata=_usage_meta(chunk.usage),
                        response_metadata={
                            "model_name": chunk.model,
                            "cost_usd": chunk.usage.cost_usd,
                        },
                    )
                )


def to_langchain_tools(specs: Sequence[ToolSpec], provider: ToolProvider) -> list[BaseTool]:
    tools: list[BaseTool] = []
    for spec in specs:

        async def _run(_name: str = spec.name, **kwargs: Any) -> str:
            try:
                return await provider.call(_name, kwargs)
            except ToolExecutionError as exc:  # becomes a ToolMessage the model can react to
                raise ToolException(exc.detail) from exc

        tools.append(
            StructuredTool(
                name=spec.name,
                description=spec.description or spec.name,
                args_schema=spec.input_schema or {"type": "object", "properties": {}},
                coroutine=_run,
                handle_tool_error=True,
            )
        )
    return tools


class LangGraphRunner:
    """Streams any compiled LangGraph (plain graphs, ``create_agent``, deep agents)."""

    def __init__(
        self,
        build_graph: Callable[[list[BaseTool]], Awaitable[Any]],
        tools: ToolProvider,
        *,
        max_iterations: int = 8,
        kind_overrides: dict[str, StepKind] | None = None,
    ) -> None:
        self._build, self._tools = build_graph, tools
        self._max_iterations = max_iterations
        self._overrides = kind_overrides or {}
        self._graph: Any = None
        self._specs: dict[str, ToolSpec] = {}

    async def _ensure(self) -> Any:
        if self._graph is None:  # tools (esp. MCP) are only known once servers have connected
            specs = await self._tools.list_tools()
            self._specs = {s.name: s for s in specs}
            self._graph = await self._build(to_langchain_tools(specs, self._tools))
        return self._graph

    def _kind(self, name: str) -> StepKind:
        return self._overrides.get(name) or step_kind_for(self._specs.get(name))

    async def stream(self, request: AgentRequest) -> AsyncIterator[AgentEvent]:
        from langgraph.errors import GraphRecursionError

        rid = request.run_id
        graph = await self._ensure()
        yield RunStarted(
            run_id=rid,
            thread_id=request.thread_id,
            agent_name=request.agent_name,
            agent_version=request.agent_version,
        )
        inputs = {"messages": [(m.role, m.content) for m in request.messages]}
        config = {
            "run_name": f"{request.agent_name}@{request.agent_version}",
            "metadata": {"thread_id": request.thread_id, "run_id": rid},
            "recursion_limit": self._max_iterations * 2 + 1,
        }
        started: dict[str, float] = {}
        final_parts: list[str] = []
        total = Usage()

        def parent_of(ev: dict[str, Any]) -> str | None:
            for pid in reversed(ev.get("parent_ids", [])):
                if pid in started:
                    return str(pid)
            return None

        try:
            async for ev in graph.astream_events(inputs, version="v2", config=config):
                kind, name, lc_id = ev["event"], ev.get("name", ""), str(ev.get("run_id", ""))
                data = ev.get("data", {})
                if kind == "on_chat_model_stream":
                    chunk = data["chunk"]
                    # Text streamed alongside a tool call is "thinking", shown only as final text
                    if isinstance(chunk.content, str) and chunk.content and not started:
                        final_parts.append(chunk.content)
                        yield MessageDelta(run_id=rid, text=chunk.content)
                elif kind == "on_chat_model_end":
                    out = data.get("output")
                    meta = getattr(out, "usage_metadata", None) or {}
                    if meta:
                        rmeta = getattr(out, "response_metadata", {}) or {}
                        usage = Usage(
                            meta.get("input_tokens", 0),
                            meta.get("output_tokens", 0),
                            float(rmeta.get("cost_usd", 0.0)),
                        )
                        total = total + usage
                        yield UsageReported(
                            run_id=rid, usage=usage, model=str(rmeta.get("model_name", ""))
                        )
                elif kind in ("on_tool_start", "on_retriever_start"):
                    started[lc_id] = time.perf_counter()
                    yield StepStarted(
                        run_id=rid,
                        step_id=lc_id,
                        kind=StepKind.RAG if kind == "on_retriever_start" else self._kind(name),
                        name=name,
                        parent_id=parent_of(ev),
                        input={"input": summarize(data.get("input", ""), 300)},
                    )
                elif kind in ("on_tool_end", "on_retriever_end") and lc_id in started:
                    out = data.get("output", "")
                    yield StepFinished(
                        run_id=rid,
                        step_id=lc_id,
                        # handled tool errors arrive as a normal end event with status="error"
                        ok=getattr(out, "status", "success") != "error",
                        output_summary=summarize(getattr(out, "content", out), 300),
                        duration_ms=int((time.perf_counter() - started.pop(lc_id)) * 1000),
                    )
                elif kind in ("on_tool_error", "on_retriever_error") and lc_id in started:
                    yield StepFinished(
                        run_id=rid,
                        step_id=lc_id,
                        ok=False,
                        output_summary=summarize(data.get("error", ""), 300),
                        duration_ms=int((time.perf_counter() - started.pop(lc_id)) * 1000),
                    )
        except GraphRecursionError as exc:
            raise IterationLimitError(
                f"agent exceeded {self._max_iterations} iterations without finishing"
            ) from exc
        yield MessageCompleted(run_id=rid, text="".join(final_parts))
        yield RunCompleted(run_id=rid, usage=total)
