"""Async circuit breaker: CLOSED -> OPEN (after N consecutive failures) -> HALF_OPEN (probe)."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from enum import IntEnum

import structlog

from agent_svc.domain.exceptions import (
    CircuitOpenError,
    UnavailableError,
    UpstreamError,
    UpstreamTimeoutError,
)
from agent_svc.observability.metrics import BREAKER_STATE, BREAKER_TRANSITIONS

log = structlog.get_logger()


class BreakerState(IntEnum):
    CLOSED = 0
    HALF_OPEN = 1
    OPEN = 2


def default_is_failure(exc: BaseException) -> bool:
    """Only dependency faults trip the breaker, never caller errors or cancellation."""
    return isinstance(exc, UpstreamError | UpstreamTimeoutError) or (
        isinstance(exc, UnavailableError) and not isinstance(exc, CircuitOpenError)
    )


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        reset_seconds: float = 30.0,
        is_failure: Callable[[BaseException], bool] = default_is_failure,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self._threshold = failure_threshold
        self._reset = reset_seconds
        self._is_failure = is_failure
        self._clock = clock
        self._state = BreakerState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._probe_in_flight = False
        BREAKER_STATE.set(0, {"name": name})

    @property
    def state(self) -> BreakerState:
        if self._state is BreakerState.OPEN and self._clock() - self._opened_at >= self._reset:
            self._transition(BreakerState.HALF_OPEN)
        return self._state

    def _transition(self, to: BreakerState) -> None:
        if to is self._state:
            return
        log.warning("breaker_transition", breaker=self.name, to=to.name)
        self._state = to
        BREAKER_STATE.set(int(to), {"name": self.name})
        BREAKER_TRANSITIONS.add(1, {"name": self.name, "to": to.name.lower()})

    def _acquire(self) -> bool:
        """Returns True if this call is the half-open probe."""
        state = self.state
        if state is BreakerState.OPEN:
            raise CircuitOpenError(f"circuit {self.name!r} is open")
        if state is BreakerState.HALF_OPEN:
            if self._probe_in_flight:
                raise CircuitOpenError(f"circuit {self.name!r} is half-open (probe in flight)")
            self._probe_in_flight = True
            return True
        return False

    def _success(self, probe: bool) -> None:
        if probe:
            self._probe_in_flight = False
        self._failures = 0
        self._transition(BreakerState.CLOSED)

    def _failure(self, probe: bool) -> None:
        if probe:
            self._probe_in_flight = False
            self._opened_at = self._clock()
            self._transition(BreakerState.OPEN)
            return
        self._failures += 1
        if self._failures >= self._threshold:
            self._opened_at = self._clock()
            self._transition(BreakerState.OPEN)

    @asynccontextmanager
    async def guard(self) -> AsyncIterator[None]:
        probe = self._acquire()
        try:
            yield
        except BaseException as exc:
            if self._is_failure(exc):
                self._failure(probe)
            elif probe:  # caller error or cancellation: release the probe slot only
                self._probe_in_flight = False
            raise
        else:
            self._success(probe)

    async def call[T](self, fn: Callable[[], Awaitable[T]]) -> T:
        async with self.guard():
            return await fn()
