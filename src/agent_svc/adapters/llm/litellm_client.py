"""LiteLLM-backed LLMClient: one adapter for 100+ providers; maps errors to domain errors."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any

from agent_svc.domain.exceptions import (
    BadRequestError,
    DomainError,
    UpstreamError,
    UpstreamTimeoutError,
)
from agent_svc.domain.models import Message, ToolCall, Usage
from agent_svc.domain.ports import LLMChunk, LLMResponse

Completion = Callable[..., Awaitable[Any]]


def map_litellm_error(exc: Exception) -> DomainError:
    import litellm.exceptions as lex

    if isinstance(exc, lex.Timeout):
        return UpstreamTimeoutError(str(exc))
    if isinstance(exc, lex.BadRequestError | lex.ContentPolicyViolationError):
        return BadRequestError(f"LLM rejected the request: {exc}")
    if isinstance(exc, lex.AuthenticationError | lex.NotFoundError):
        return UpstreamError("LLM provider credentials/model misconfigured", retryable=False)
    return UpstreamError(str(exc))  # rate limit, 5xx, connection errors: retryable


def _usage(raw: Any, model: str) -> Usage:
    if raw is None:
        return Usage()
    cost = 0.0
    try:
        import litellm

        cost = float(litellm.completion_cost(completion_response=raw) or 0.0)
    except Exception:
        cost = 0.0
    u = getattr(raw, "usage", None) or raw
    return Usage(
        int(getattr(u, "prompt_tokens", 0) or 0), int(getattr(u, "completion_tokens", 0) or 0), cost
    )


def _args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_call(c: Any) -> ToolCall:
    return ToolCall(str(c.id), str(c.function.name), _args(c.function.arguments))


def _finish_calls(partial: dict[int, dict[str, str]]) -> tuple[ToolCall, ...]:
    return tuple(
        ToolCall(v["id"] or f"call_{i}", v["name"], _args(v["args"]))
        for i, v in sorted(partial.items())
    )


class LiteLLMClient:
    def __init__(
        self, model: str, *, timeout: float = 60.0, completion: Completion | None = None
    ) -> None:
        self._model = model
        self._timeout = timeout
        self._completion = completion

    def _fn(self) -> Completion:
        if self._completion is not None:
            return self._completion
        import litellm

        fn: Completion = litellm.acompletion
        return fn

    @staticmethod
    def _payload(messages: Sequence[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            d: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                    }
                    for c in m.tool_calls
                ]
            if m.tool_call_id:
                d["tool_call_id"] = m.tool_call_id
            if m.name and m.role == "tool":
                d["name"] = m.name
            out.append(d)
        return out

    async def complete(
        self, messages: Sequence[Message], *, model: str | None = None, **params: Any
    ) -> LLMResponse:
        used = model or self._model
        try:
            resp = await self._fn()(
                model=used, messages=self._payload(messages), timeout=self._timeout, **params
            )
        except DomainError:
            raise
        except Exception as exc:
            raise map_litellm_error(exc) from exc
        msg = resp.choices[0].message
        calls = tuple(_parse_call(c) for c in (getattr(msg, "tool_calls", None) or []))
        return LLMResponse(
            text=msg.content or "", model=used, usage=_usage(resp, used), tool_calls=calls
        )

    async def stream(
        self, messages: Sequence[Message], *, model: str | None = None, **params: Any
    ) -> AsyncIterator[LLMChunk]:
        used = model or self._model
        try:
            resp = await self._fn()(
                model=used,
                messages=self._payload(messages),
                timeout=self._timeout,
                stream=True,
                stream_options={"include_usage": True},
                **params,
            )
            partial: dict[int, dict[str, str]] = {}  # tool-call deltas arrive in fragments
            async for part in resp:
                delta = ""
                if part.choices:
                    d = part.choices[0].delta
                    delta = d.content or ""
                    for tc in getattr(d, "tool_calls", None) or []:
                        slot = partial.setdefault(
                            int(getattr(tc, "index", 0) or 0), {"id": "", "name": "", "args": ""}
                        )
                        slot["id"] = getattr(tc, "id", None) or slot["id"]
                        fn = getattr(tc, "function", None)
                        if fn is not None:
                            slot["name"] += getattr(fn, "name", None) or ""
                            slot["args"] += getattr(fn, "arguments", None) or ""
                usage = _usage(part, used) if getattr(part, "usage", None) else None
                if usage:  # emit accumulated tool calls before the closing usage chunk
                    if partial:
                        yield LLMChunk(tool_calls=_finish_calls(partial), model=used)
                        partial = {}
                    yield LLMChunk(delta=delta, usage=usage, model=used)
                elif delta:
                    yield LLMChunk(delta=delta, model=used)
            if partial:
                yield LLMChunk(tool_calls=_finish_calls(partial), model=used)
        except DomainError:
            raise
        except Exception as exc:
            raise map_litellm_error(exc) from exc


def build(settings: Any, model: str) -> LiteLLMClient:
    return LiteLLMClient(model, timeout=settings.llm.timeout_seconds)
