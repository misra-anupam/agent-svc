"""LangGraph adapter.

* ``PortChatModel`` exposes our resilient ``LLMClient`` port as a LangChain chat model, so
  breaker/retry/fallback apply to anything built on LangGraph.
* ``LangGraphRunner`` maps ``astream_events`` (v2) to framework-neutral ``AgentEvent``s.
* ``build`` creates the reference graph; replace ``build_graph`` with your own agent.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import ConfigDict

from agent_svc.adapters.orchestrators._text import summarize
from agent_svc.config.settings import Settings
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
    Usage,
    UsageReported,
)
from agent_svc.domain.ports import LLMClient

_ROLES: dict[str, Role] = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}


def _to_domain(msgs: list[BaseMessage]) -> list[Message]:
    return [Message(_ROLES.get(m.type, "user"), str(m.content)) for m in msgs]


class PortChatModel(BaseChatModel):
    """LangChain chat model backed by the LLMClient port (async only)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm: Any

    @property
    def _llm_type(self) -> str:
        return "agent-svc-port"

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise NotImplementedError("PortChatModel is async-only; use ainvoke/astream")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        resp = await self.llm.complete(_to_domain(messages))
        u = resp.usage
        msg = AIMessage(
            content=resp.text,
            usage_metadata={
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "total_tokens": u.total_tokens,
            },
            response_metadata={"model_name": resp.model, "cost_usd": u.cost_usd},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        async for chunk in self.llm.stream(_to_domain(messages)):
            if chunk.delta:
                if run_manager:
                    await run_manager.on_llm_new_token(chunk.delta)
                yield ChatGenerationChunk(message=AIMessageChunk(content=chunk.delta))
            if chunk.usage:
                u = chunk.usage
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        usage_metadata={
                            "input_tokens": u.input_tokens,
                            "output_tokens": u.output_tokens,
                            "total_tokens": u.total_tokens,
                        },
                        response_metadata={"model_name": chunk.model, "cost_usd": u.cost_usd},
                    )
                )


def build_graph(model: BaseChatModel) -> Any:
    async def respond(state: MessagesState) -> dict[str, list[BaseMessage]]:
        return {"messages": [await model.ainvoke(state["messages"])]}

    g = StateGraph(MessagesState)
    g.add_node("respond", respond)
    g.add_edge(START, "respond")
    g.add_edge("respond", END)
    return g.compile()


_NODE_NOISE = {"LangGraph", "__start__", "__end__"}


class LangGraphRunner:
    def __init__(self, graph: Any) -> None:
        self._graph = graph

    async def stream(self, request: AgentRequest) -> AsyncIterator[AgentEvent]:
        rid = request.run_id
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
        }
        started: dict[str, float] = {}  # lc run_id -> start time, for emitted steps
        text_parts: list[str] = []
        total = Usage()

        def parent_of(ev: dict[str, Any]) -> str | None:
            for pid in reversed(ev.get("parent_ids", [])):
                if pid in started:
                    return str(pid)
            return None

        async for ev in self._graph.astream_events(inputs, version="v2", config=config):
            kind, name, lc_id = ev["event"], ev.get("name", ""), str(ev.get("run_id", ""))
            data = ev.get("data", {})
            if kind == "on_chat_model_stream":
                content = data["chunk"].content
                if isinstance(content, str) and content:
                    text_parts.append(content)
                    yield MessageDelta(run_id=rid, text=content)
            elif kind == "on_chat_model_end":
                meta = getattr(data.get("output"), "usage_metadata", None) or {}
                if meta:
                    resp_meta = getattr(data["output"], "response_metadata", {}) or {}
                    usage = Usage(
                        meta.get("input_tokens", 0),
                        meta.get("output_tokens", 0),
                        float(resp_meta.get("cost_usd", 0.0)),
                    )
                    total = total + usage
                    yield UsageReported(
                        run_id=rid, usage=usage, model=str(resp_meta.get("model_name", ""))
                    )
            elif kind in ("on_tool_start", "on_retriever_start") or (
                kind == "on_chain_start"
                and ev.get("metadata", {}).get("langgraph_node") == name
                and name not in _NODE_NOISE
            ):
                step_kind = (
                    StepKind.TOOL
                    if kind == "on_tool_start"
                    else StepKind.RAG
                    if kind == "on_retriever_start"
                    else StepKind.AGENT
                )
                started[lc_id] = time.perf_counter()
                yield StepStarted(
                    run_id=rid,
                    step_id=lc_id,
                    kind=step_kind,
                    name=name,
                    parent_id=parent_of(ev),
                    input=(
                        {}
                        if step_kind is StepKind.AGENT  # node state is noisy; tools/RAG show args
                        else {"input": summarize(data.get("input", ""), 300)}
                    ),
                )
            elif kind in ("on_tool_end", "on_retriever_end", "on_chain_end") and lc_id in started:
                yield StepFinished(
                    run_id=rid,
                    step_id=lc_id,
                    ok=True,
                    output_summary=summarize(data.get("output", ""), 300),
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
        yield MessageCompleted(run_id=rid, text="".join(text_parts))
        yield RunCompleted(run_id=rid, usage=total)


def build(_settings: Settings, llm: LLMClient) -> LangGraphRunner:
    return LangGraphRunner(build_graph(PortChatModel(llm=llm)))
