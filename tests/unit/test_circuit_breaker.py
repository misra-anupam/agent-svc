import pytest

from agent_svc.adapters.resilience.circuit_breaker import BreakerState, CircuitBreaker
from agent_svc.domain.exceptions import BadRequestError, CircuitOpenError, UpstreamError


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


async def _fail(cb: CircuitBreaker, exc: Exception | None = None) -> None:
    exc = exc or UpstreamError("x")
    with pytest.raises(type(exc)):
        async with cb.guard():
            raise exc


async def test_opens_after_threshold_and_fails_fast() -> None:
    cb = CircuitBreaker("t1", failure_threshold=3, reset_seconds=10, clock=Clock())
    for _ in range(3):
        await _fail(cb)
    assert cb.state is BreakerState.OPEN
    with pytest.raises(CircuitOpenError):
        async with cb.guard():
            pytest.fail("must not run")


async def test_half_open_probe_closes_on_success() -> None:
    clock = Clock()
    cb = CircuitBreaker("t2", failure_threshold=1, reset_seconds=10, clock=clock)
    await _fail(cb)
    clock.t = 11
    assert cb.state is BreakerState.HALF_OPEN
    async with cb.guard():
        pass
    assert cb.state is BreakerState.CLOSED


async def test_half_open_probe_failure_reopens_and_blocks_second_probe() -> None:
    clock = Clock()
    cb = CircuitBreaker("t3", failure_threshold=1, reset_seconds=10, clock=clock)
    await _fail(cb)
    clock.t = 11
    with pytest.raises(UpstreamError):  # probe fails
        async with cb.guard():
            raise UpstreamError("still down")
    assert cb.state is BreakerState.OPEN


async def test_only_one_probe_in_flight() -> None:
    clock = Clock()
    cb = CircuitBreaker("t4", failure_threshold=1, reset_seconds=10, clock=clock)
    await _fail(cb)
    clock.t = 11
    async with cb.guard():
        with pytest.raises(CircuitOpenError):
            async with cb.guard():
                pass


async def test_caller_errors_do_not_trip_breaker() -> None:
    cb = CircuitBreaker("t5", failure_threshold=1, clock=Clock())
    await _fail(cb, BadRequestError("bad"))
    assert cb.state is BreakerState.CLOSED


async def test_success_resets_failure_count() -> None:
    cb = CircuitBreaker("t6", failure_threshold=2, clock=Clock())
    await _fail(cb)
    async with cb.guard():
        pass
    await _fail(cb)
    assert cb.state is BreakerState.CLOSED
