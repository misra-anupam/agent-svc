from types import SimpleNamespace as NS

import litellm.exceptions as lex
import pytest

from agent_svc.adapters.llm.litellm_client import LiteLLMClient
from agent_svc.domain.exceptions import BadRequestError, UpstreamError, UpstreamTimeoutError
from agent_svc.domain.models import Message

MSGS = [Message("user", "hi")]


def _resp(text: str) -> NS:
    return NS(
        choices=[NS(message=NS(content=text))], usage=NS(prompt_tokens=3, completion_tokens=2)
    )


async def test_complete_maps_response_and_usage() -> None:
    async def fake(**kw):  # type: ignore[no-untyped-def]
        assert kw["messages"] == [{"role": "user", "content": "hi"}]
        return _resp("yo")

    r = await LiteLLMClient("m/x", completion=fake).complete(MSGS)
    assert (r.text, r.model, r.usage.input_tokens, r.usage.output_tokens) == ("yo", "m/x", 3, 2)


@pytest.mark.parametrize(
    ("raised", "expected", "retryable"),
    [
        (lex.Timeout("t", model="m", llm_provider="p"), UpstreamTimeoutError, True),
        (lex.RateLimitError("r", model="m", llm_provider="p"), UpstreamError, True),
        (lex.AuthenticationError("a", model="m", llm_provider="p"), UpstreamError, False),
        (lex.BadRequestError("b", model="m", llm_provider="p"), BadRequestError, False),
    ],
)
async def test_error_mapping(raised: Exception, expected: type, retryable: bool) -> None:
    async def boom(**kw):  # type: ignore[no-untyped-def]
        raise raised

    with pytest.raises(expected) as ei:
        await LiteLLMClient("m/x", completion=boom).complete(MSGS)
    assert ei.value.retryable is retryable  # type: ignore[attr-defined]


async def test_stream_yields_deltas_then_usage() -> None:
    async def gen():  # type: ignore[no-untyped-def]
        yield NS(choices=[NS(delta=NS(content="he"))], usage=None)
        yield NS(choices=[NS(delta=NS(content="llo"))], usage=None)
        yield NS(choices=[], usage=NS(prompt_tokens=1, completion_tokens=2))

    async def fake(**kw):  # type: ignore[no-untyped-def]
        assert kw["stream"] is True
        return gen()

    chunks = [c async for c in LiteLLMClient("m/x", completion=fake).stream(MSGS)]
    assert "".join(c.delta for c in chunks) == "hello"
    assert chunks[-1].usage is not None and chunks[-1].usage.output_tokens == 2
