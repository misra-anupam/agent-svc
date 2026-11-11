"""Composable LLMClient decorators: Breaker, Retrying, Fallback. All implement the LLMClient port.

Streaming rule: retries and fallback only happen *before the first chunk* is emitted; once text
has reached the client we cannot transparently switch providers, so the error propagates.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

import structlog
from tenacity import AsyncRetrying

from agent_svc.adapters.resilience.circuit_breaker import CircuitBreaker
from agent_svc.domain.exceptions import CircuitOpenError, DomainError
from agent_svc.domain.models import Message
from agent_svc.domain.ports import LLMChunk, LLMClient, LLMResponse
from agent_svc.observability.metrics import LLM_CALLS, LLM_FALLBACKS, LLM_TOKENS

log = structlog.get_logger()
_END = object()


class BreakerLLM:
    def __init__(self, inner: LLMClient, breaker: CircuitBreaker) -> None:
        self._inner, self._breaker = inner, breaker

    async def complete(
        self, messages: Sequence[Message], *, model: str | None = None, **params: Any
    ) -> LLMResponse:
        return await self._breaker.call(
            lambda: self._inner.complete(messages, model=model, **params)
        )

    async def stream(
        self, messages: Sequence[Message], *, model: str | None = None, **params: Any
    ) -> AsyncIterator[LLMChunk]:
        async with self._breaker.guard():
            async for chunk in self._inner.stream(messages, model=model, **params):
                yield chunk


class RetryingLLM:
    def __init__(self, inner: LLMClient, retrying: Callable[[], AsyncRetrying]) -> None:
        self._inner, self._retrying = inner, retrying

    async def complete(
        self, messages: Sequence[Message], *, model: str | None = None, **params: Any
    ) -> LLMResponse:
        async for attempt in self._retrying():
            with attempt:
                return await self._inner.complete(messages, model=model, **params)
        raise AssertionError("unreachable")  # pragma: no cover

    async def stream(
        self, messages: Sequence[Message], *, model: str | None = None, **params: Any
    ) -> AsyncIterator[LLMChunk]:
        it: AsyncIterator[LLMChunk] | None = None
        first: Any = _END
        async for attempt in self._retrying():
            with attempt:
                it = self._inner.stream(messages, model=model, **params).__aiter__()
                first = await anext(it, _END)
        if first is _END or it is None:
            return
        yield first
        async for chunk in it:
            yield chunk


def _should_fallback(exc: BaseException) -> bool:
    return isinstance(exc, DomainError) and (exc.retryable or isinstance(exc, CircuitOpenError))


class FallbackLLM:
    """Primary first; on a provider failure, transparently use the secondary."""

    def __init__(
        self, primary: LLMClient, secondary: LLMClient, *, primary_name: str = "primary"
    ) -> None:
        self._primary, self._secondary, self._name = primary, secondary, primary_name

    async def complete(
        self, messages: Sequence[Message], *, model: str | None = None, **params: Any
    ) -> LLMResponse:
        try:
            resp = await self._primary.complete(messages, model=model, **params)
        except DomainError as exc:
            if not _should_fallback(exc):
                raise
            self._note_fallback(exc)
            resp = await self._secondary.complete(messages, **params)
        return resp

    async def stream(
        self, messages: Sequence[Message], *, model: str | None = None, **params: Any
    ) -> AsyncIterator[LLMChunk]:
        emitted = False
        try:
            async for chunk in self._primary.stream(messages, model=model, **params):
                emitted = True
                yield chunk
            return
        except DomainError as exc:
            if emitted or not _should_fallback(exc):
                raise
            self._note_fallback(exc)
        async for chunk in self._secondary.stream(messages, **params):
            yield chunk

    def _note_fallback(self, exc: DomainError) -> None:
        LLM_FALLBACKS.add(1, {"from_model": self._name})
        log.warning("llm_fallback", primary=self._name, reason=exc.code)


class MeteredLLM:
    """Records call outcome and token metrics."""

    def __init__(self, inner: LLMClient, name: str) -> None:
        self._inner, self._name = inner, name

    async def complete(
        self, messages: Sequence[Message], *, model: str | None = None, **params: Any
    ) -> LLMResponse:
        try:
            resp = await self._inner.complete(messages, model=model, **params)
        except BaseException:
            LLM_CALLS.add(1, {"model": self._name, "outcome": "error"})
            raise
        LLM_CALLS.add(1, {"model": resp.model or self._name, "outcome": "ok"})
        LLM_TOKENS.add(
            resp.usage.input_tokens, {"model": resp.model or self._name, "direction": "input"}
        )
        LLM_TOKENS.add(
            resp.usage.output_tokens, {"model": resp.model or self._name, "direction": "output"}
        )
        return resp

    async def stream(
        self, messages: Sequence[Message], *, model: str | None = None, **params: Any
    ) -> AsyncIterator[LLMChunk]:
        try:
            async for chunk in self._inner.stream(messages, model=model, **params):
                if chunk.usage:
                    m = chunk.model or self._name
                    LLM_TOKENS.add(chunk.usage.input_tokens, {"model": m, "direction": "input"})
                    LLM_TOKENS.add(chunk.usage.output_tokens, {"model": m, "direction": "output"})
                yield chunk
        except BaseException:
            LLM_CALLS.add(1, {"model": self._name, "outcome": "error"})
            raise
        LLM_CALLS.add(1, {"model": self._name, "outcome": "ok"})
