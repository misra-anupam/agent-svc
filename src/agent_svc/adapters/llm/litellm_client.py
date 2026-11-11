"""LiteLLM-backed LLMClient: one adapter for 100+ providers; maps errors to domain errors."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any

from agent_svc.domain.exceptions import (
    BadRequestError,
    DomainError,
    UpstreamError,
    UpstreamTimeoutError,
)
from agent_svc.domain.models import Message, Usage
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
    def _payload(messages: Sequence[Message]) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

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
        text = resp.choices[0].message.content or ""
        return LLMResponse(text=text, model=used, usage=_usage(resp, used))

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
            async for part in resp:
                delta = ""
                if part.choices:
                    delta = part.choices[0].delta.content or ""
                usage = _usage(part, used) if getattr(part, "usage", None) else None
                if delta or usage:
                    yield LLMChunk(delta=delta, usage=usage, model=used)
        except DomainError:
            raise
        except Exception as exc:
            raise map_litellm_error(exc) from exc


def build(settings: Any, model: str) -> LiteLLMClient:
    return LiteLLMClient(model, timeout=settings.llm.timeout_seconds)
