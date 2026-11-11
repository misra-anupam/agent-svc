"""Google ADK adapter.

* ``PortAdkLlm``  - an ADK ``BaseLlm`` backed by our resilient ``LLMClient`` port, so ADK agents get
  retry / circuit breaker / primary->secondary fallback / metering like every other framework.
* ``ProviderTool`` - exposes any ``ToolProvider`` tool (local, MCP) as an ADK tool.
* ``AdkRunner``    - maps ADK ``Event``s to framework-neutral ``AgentEvent``s.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.events.event import Event
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.base_tool import BaseTool
from google.genai import types
from pydantic import ConfigDict

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

_APP = "agent-svc"


# --- ADK <-> port translation ----------------------------------------------------------------


def _lower_types(schema: Any) -> Any:
    """genai Schema uses upper-case type names ('OBJECT'); JSON Schema wants lower-case."""
    if isinstance(schema, dict):
        return {
            k: (v.lower() if k == "type" and isinstance(v, str) else _lower_types(v))
            for k, v in schema.items()
        }
    if isinstance(schema, list):
        return [_lower_types(v) for v in schema]
    return schema


def _openai_tools(req: LlmRequest) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tool in (req.config.tools if req.config else None) or []:
        for d in getattr(tool, "function_declarations", None) or []:
            if d.parameters_json_schema:
                params = d.parameters_json_schema
            elif d.parameters:
                params = _lower_types(d.parameters.model_dump(exclude_none=True, mode="json"))
            else:
                params = {"type": "object", "properties": {}}
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": d.name,
                        "description": d.description or d.name,
                        "parameters": params,
                    },
                }
            )
    return out


def _instruction_text(req: LlmRequest) -> str:
    si = req.config.system_instruction if req.config else None
    if si is None:
        return ""
    if isinstance(si, str):
        return si
    parts = getattr(si, "parts", None) or []
    return "\n".join(p.text for p in parts if getattr(p, "text", None))


def _messages(req: LlmRequest) -> list[Message]:
    msgs: list[Message] = []
    if text := _instruction_text(req):
        msgs.append(Message("system", text))
    for content in req.contents or []:
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for p in content.parts or []:
            if p.function_response is not None:
                fr = p.function_response
                resp = fr.response or {}
                body = resp.get("result", resp.get("error", resp))
                msgs.append(Message("tool", str(body), name=fr.name, tool_call_id=fr.id or fr.name))
            elif p.function_call is not None:
                fc = p.function_call
                calls.append(
                    ToolCall(fc.id or f"call_{fc.name}", fc.name or "", dict(fc.args or {}))
                )
            elif p.text and not getattr(p, "thought", False):
                text_parts.append(p.text)
        if text_parts or calls:
            role: Role = "assistant" if content.role == "model" else "user"
            msgs.append(Message(role, "".join(text_parts), tool_calls=tuple(calls)))
    return msgs


def _usage_meta(u: Usage) -> types.GenerateContentResponseUsageMetadata:
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=u.input_tokens,
        candidates_token_count=u.output_tokens,
        total_token_count=u.total_tokens,
    )


def _final_response(
    text: str, calls: tuple[ToolCall, ...], usage: Usage, model: str
) -> LlmResponse:
    parts: list[types.Part] = []
    if text:
        parts.append(types.Part(text=text))
    parts.extend(
        types.Part(function_call=types.FunctionCall(id=c.id, name=c.name, args=c.arguments))
        for c in calls
    )
    return LlmResponse(
        content=types.Content(role="model", parts=parts),
        usage_metadata=_usage_meta(usage),
        model_version=model,
        custom_metadata={"cost_usd": usage.cost_usd, "model": model},
        partial=False,
    )


class PortAdkLlm(BaseLlm):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm: Any

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        messages = _messages(llm_request)
        tools = _openai_tools(llm_request)
        params: dict[str, Any] = {"tools": tools} if tools else {}
        if not stream:
            r = await self.llm.complete(messages, **params)
            yield _final_response(r.text, r.tool_calls, r.usage, r.model)
            return
        text: list[str] = []
        calls: tuple[ToolCall, ...] = ()
        usage, model = Usage(), ""
        async for chunk in self.llm.stream(messages, **params):
            model = chunk.model or model
            if chunk.delta:
                text.append(chunk.delta)
                yield LlmResponse(
                    content=types.Content(role="model", parts=[types.Part(text=chunk.delta)]),
                    partial=True,
                )
            calls = chunk.tool_calls or calls
            usage = chunk.usage or usage
        yield _final_response("".join(text), calls, usage, model)


class ProviderTool(BaseTool):
    def __init__(self, spec: ToolSpec, provider: ToolProvider) -> None:
        super().__init__(name=spec.name, description=spec.description or spec.name)
        self._spec, self._provider = spec, provider

    def _get_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self._spec.name,
            description=self._spec.description or self._spec.name,
            parameters_json_schema=self._spec.input_schema or {"type": "object", "properties": {}},
        )

    async def run_async(self, *, args: dict[str, Any], tool_context: Any) -> dict[str, Any]:
        try:
            return {"result": await self._provider.call(self._spec.name, args)}
        except ToolExecutionError as exc:  # the model sees the error and can recover
            return {"error": exc.detail}


# --- runner ----------------------------------------------------------------------------------


class AdkRunner:
    def __init__(self, settings: Settings, llm: LLMClient, tools: ToolProvider) -> None:
        self._settings, self._llm, self._tools = settings, llm, tools
        self._runner: Runner | None = None
        self._sessions = InMemorySessionService()
        self._specs: dict[str, ToolSpec] = {}

    async def _ensure(self) -> Runner:
        if self._runner is None:
            specs = await self._tools.list_tools()
            self._specs = {s.name: s for s in specs}
            base = self._settings.orchestrator.system_prompt

            def instruction(ctx: Any) -> str:  # memory context rides in session state
                extra = ctx.state.get("memory_context", "")
                return f"{base}\n\n{extra}" if extra else base

            agent = LlmAgent(
                name="agent",
                model=PortAdkLlm(model="agent-svc-port", llm=self._llm),
                instruction=instruction,
                tools=[ProviderTool(s, self._tools) for s in specs],
            )
            self._runner = Runner(
                app_name=_APP,
                agent=agent,
                session_service=self._sessions,
                auto_create_session=False,
            )
        return self._runner

    async def stream(self, request: AgentRequest) -> AsyncIterator[AgentEvent]:
        rid = request.run_id
        runner = await self._ensure()
        yield RunStarted(
            run_id=rid,
            thread_id=request.thread_id,
            agent_name=request.agent_name,
            agent_version=request.agent_version,
        )
        system = "\n".join(m.content for m in request.messages if m.role == "system")
        convo = [m for m in request.messages if m.role != "system"]
        user_id = request.principal.subject
        session = await self._sessions.create_session(
            app_name=_APP, user_id=user_id, session_id=rid, state={"memory_context": system}
        )
        for m in convo[:-1]:  # replay prior turns; the last user message is the new input
            await self._sessions.append_event(
                session,
                Event(
                    author="user" if m.role == "user" else "agent",
                    content=types.Content(
                        role="user" if m.role == "user" else "model",
                        parts=[types.Part(text=m.content)],
                    ),
                ),
            )
        new_message = types.Content(role="user", parts=[types.Part(text=convo[-1].content)])

        started: dict[str, float] = {}
        text_out: list[str] = []
        seen_partial = False
        total = Usage()
        try:
            async for ev in runner.run_async(
                user_id=user_id,
                session_id=rid,
                new_message=new_message,
                run_config=RunConfig(
                    streaming_mode=StreamingMode.SSE,
                    max_llm_calls=self._settings.orchestrator.max_iterations,
                ),
            ):
                if ev.author == "user":
                    continue
                parts = (ev.content.parts or []) if ev.content else []
                text = "".join(p.text for p in parts if p.text)
                if ev.partial:
                    if text:
                        seen_partial = True
                        text_out.append(text)
                        yield MessageDelta(run_id=rid, text=text)
                    continue
                if text and not seen_partial:
                    text_out.append(text)
                    yield MessageDelta(run_id=rid, text=text)
                seen_partial = False
                if ev.usage_metadata and (
                    ev.get_function_calls() or text or ev.is_final_response()
                ):
                    meta = ev.custom_metadata or {}
                    usage = Usage(
                        ev.usage_metadata.prompt_token_count or 0,
                        ev.usage_metadata.candidates_token_count or 0,
                        float(meta.get("cost_usd", 0.0)),
                    )
                    total = total + usage
                    yield UsageReported(
                        run_id=rid,
                        usage=usage,
                        model=str(meta.get("model") or ev.model_version or ""),
                    )
                for fc in ev.get_function_calls():
                    sid = fc.id or uuid.uuid4().hex
                    started[sid] = time.perf_counter()
                    yield StepStarted(
                        run_id=rid,
                        step_id=sid,
                        kind=step_kind_for(self._specs.get(fc.name or "")),
                        name=fc.name or "",
                        input={"input": summarize(dict(fc.args or {}), 300)},
                    )
                for fr in ev.get_function_responses():
                    sid = fr.id or ""
                    if sid not in started:
                        continue
                    resp = fr.response or {}
                    yield StepFinished(
                        run_id=rid,
                        step_id=sid,
                        ok="error" not in resp,
                        output_summary=summarize(resp.get("result", resp.get("error", "")), 300),
                        duration_ms=int((time.perf_counter() - started.pop(sid)) * 1000),
                    )
        except LlmCallsLimitExceededError as exc:
            raise IterationLimitError(
                f"agent exceeded {self._settings.orchestrator.max_iterations} model calls"
            ) from exc
        finally:
            await self._sessions.delete_session(app_name=_APP, user_id=user_id, session_id=rid)
        yield MessageCompleted(run_id=rid, text="".join(text_out))
        yield RunCompleted(run_id=rid, usage=total)


def build(settings: Settings, llm: LLMClient, tools: ToolProvider) -> AdkRunner:
    return AdkRunner(settings, llm, tools)
