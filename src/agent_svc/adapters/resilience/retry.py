"""tenacity policy: exponential backoff + jitter, only for retryable domain errors."""

from __future__ import annotations

import logging

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from agent_svc.domain.exceptions import DomainError
from agent_svc.observability.metrics import LLM_RETRIES

_stdlib_log = logging.getLogger("agent_svc.retry")


def is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, DomainError) and exc.retryable


def make_retrying(
    provider: str, *, max_attempts: int = 3, initial: float = 0.5, max_wait: float = 8.0
) -> AsyncRetrying:
    def _count(state: RetryCallState) -> None:
        LLM_RETRIES.labels(provider).inc()

    log_sleep = before_sleep_log(_stdlib_log, logging.WARNING)

    def before_sleep(state: RetryCallState) -> None:
        _count(state)
        log_sleep(state)

    return AsyncRetrying(
        retry=retry_if_exception(is_retryable),
        stop=stop_after_attempt(max_attempts),
        # exponential backoff + jitter, spelled so it works across tenacity 9.1 and 9.2+
        wait=wait_exponential(multiplier=initial, max=max_wait) + wait_random(0, initial),
        before_sleep=before_sleep,
        reraise=True,
    )
