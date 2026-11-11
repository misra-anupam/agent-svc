import pytest

from agent_svc.adapters.llm.decorators import BreakerLLM, FallbackLLM, RetryingLLM
from agent_svc.adapters.llm.fake import FakeLLM
from agent_svc.adapters.resilience.circuit_breaker import CircuitBreaker
from agent_svc.adapters.resilience.retry import make_retrying
from agent_svc.domain.exceptions import BadRequestError, CircuitOpenError, UpstreamError
from agent_svc.domain.models import Message

MSGS = [Message("user", "hi there")]


def retrying(n: int = 3):  # type: ignore[no-untyped-def]
    return lambda: make_retrying("t", max_attempts=n, initial=0.001, max_wait=0.002)


async def test_retry_recovers_after_transient_failures() -> None:
    fake = FakeLLM(fail_times=2)
    resp = await RetryingLLM(fake, retrying()).complete(MSGS)
    assert resp.text and fake.calls == 3


async def test_retry_gives_up_after_max_attempts() -> None:
    fake = FakeLLM(fail_times=-1)
    with pytest.raises(UpstreamError):
        await RetryingLLM(fake, retrying(3)).complete(MSGS)
    assert fake.calls == 3


async def test_non_retryable_error_is_not_retried() -> None:
    fake = FakeLLM(fail_times=-1, error=BadRequestError("nope"))
    with pytest.raises(BadRequestError):
        await RetryingLLM(fake, retrying()).complete(MSGS)
    assert fake.calls == 1


async def test_stream_retry_before_first_chunk() -> None:
    fake = FakeLLM(reply="a b c", fail_times=1)
    chunks = [c async for c in RetryingLLM(fake, retrying()).stream(MSGS)]
    assert "".join(c.delta for c in chunks) == "a b c" and fake.calls == 2


async def test_fallback_to_secondary_on_primary_failure() -> None:
    primary, secondary = FakeLLM("p", fail_times=-1), FakeLLM("s", reply="from secondary")
    llm = FallbackLLM(primary, secondary)
    resp = await llm.complete(MSGS)
    assert resp.model == "s" and resp.text == "from secondary"


async def test_no_fallback_for_caller_errors() -> None:
    llm = FallbackLLM(FakeLLM("p", fail_times=-1, error=BadRequestError("x")), FakeLLM("s"))
    with pytest.raises(BadRequestError):
        await llm.complete(MSGS)


async def test_stream_fallback_only_before_first_chunk() -> None:
    llm = FallbackLLM(FakeLLM("p", fail_times=-1), FakeLLM("s", reply="ok now"))
    assert "".join([c.delta async for c in llm.stream(MSGS)]).strip() == "ok now"

    mid = FallbackLLM(FakeLLM("p", reply="a b c d", fail_after_chunks=2), FakeLLM("s"))
    got: list[str] = []
    with pytest.raises(UpstreamError):
        async for c in mid.stream(MSGS):
            got.append(c.delta)
    assert got  # partial text was already sent; we must not splice in another model


async def test_breaker_opens_then_fallback_serves_fast() -> None:
    primary = FakeLLM("p", fail_times=-1)
    breaker = CircuitBreaker("llm-p", failure_threshold=2, reset_seconds=60)
    hardened = RetryingLLM(BreakerLLM(primary, breaker), retrying(5))
    llm = FallbackLLM(hardened, FakeLLM("s"))
    r1 = await llm.complete(MSGS)  # 2 failures trip the breaker; retry stops on CircuitOpen
    assert r1.model == "s" and primary.calls == 2
    r2 = await llm.complete(MSGS)  # breaker open: primary not even called
    assert r2.model == "s" and primary.calls == 2
    with pytest.raises(CircuitOpenError):
        await hardened.complete(MSGS)
