import httpx
from tests.helpers import metric_total

from agent_svc.config import Settings

BODY = {"messages": [{"role": "user", "content": "hi"}]}


ROUTE = "/threads/{thread_id}/chat"


async def test_a_chat_request_records_http_llm_and_token_metrics(
    make_client, metric_reader
) -> None:  # type: ignore[no-untyped-def]
    def now() -> tuple[float, float, float]:
        return (
            metric_total(metric_reader, "agent.http.requests", route=ROUTE),
            metric_total(metric_reader, "agent.llm.calls", outcome="ok"),
            metric_total(metric_reader, "agent.llm.tokens", direction="output"),
        )

    before = now()
    async with make_client() as c:
        assert (await c.post("/v1/threads/t/chat", json=BODY)).status_code == 200
    assert all(after > was for after, was in zip(now(), before, strict=True))


async def test_the_old_metrics_endpoint_is_gone(client: httpx.AsyncClient) -> None:
    assert (await client.get("/metrics/")).status_code == 404


async def test_retry_fallback_and_breaker_metrics(make_client, metric_reader) -> None:  # type: ignore[no-untyped-def]
    from tests.scripted import ScriptedModel

    before = metric_total(metric_reader, "agent.llm.fallbacks", from_model="test/primary")
    llms = {
        "test/primary": ScriptedModel("test/primary", fail_times=-1),
        "test/secondary": ScriptedModel("test/secondary", reply="from secondary"),
    }
    async with make_client(llms) as c:
        assert (await c.post("/v1/threads/t/chat", json=BODY)).status_code == 200
    assert metric_total(metric_reader, "agent.llm.fallbacks", from_model="test/primary") > before
    assert metric_total(metric_reader, "agent.llm.retries") > 0


def test_meter_provider_builds_with_the_otel_extra() -> None:
    from agent_svc.adapters.tracing.otel_metrics import build_meter_provider

    provider = build_meter_provider(Settings(_env_file=None))  # type: ignore[call-arg]
    provider.shutdown()
