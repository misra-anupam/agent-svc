from __future__ import annotations

from collections.abc import Sequence

from tests.unit.test_cached_runner import CountingRunner, req

from agent_svc.adapters.cache.cached_runner import CachedRunner
from agent_svc.adapters.cache.semantic import SemanticIndex
from agent_svc.adapters.cache.stores import InMemoryCache
from agent_svc.config.settings import CacheSettings

# axis 0 = "refund", axis 1 = "shipping"; near-duplicates share a direction
VECTORS = {
    "how do i get a refund": [1.0, 0.0],
    "how can i get my money refunded": [0.97, 0.05],
    "where is my shipment": [0.0, 1.0],
}


class StubEmbedder:
    dim = 2

    def __init__(self) -> None:
        self.fail = False

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("embedder down")
        return [VECTORS.get(t, [0.5, 0.5]) for t in texts]


def make(
    threshold: float = 0.95, scope: str = "user"
) -> tuple[CachedRunner, CountingRunner, StubEmbedder]:
    emb, inner = StubEmbedder(), CountingRunner()
    s = CacheSettings(provider="inmemory", semantic=True, semantic_threshold=threshold, scope=scope)  # type: ignore[arg-type]
    return (
        CachedRunner(inner, InMemoryCache(), s, SemanticIndex(emb, threshold=threshold)),
        inner,
        emb,
    )


async def run(r: CachedRunner, **kw: object) -> list[str]:
    return [e.type async for e in r.stream(req(**kw))]  # type: ignore[arg-type]


async def test_a_similar_question_reuses_the_answer_without_calling_the_agent() -> None:
    r, inner, _ = make()
    await run(r, text="how do i get a refund")
    events = await run(r, text="how can i get my money refunded")
    assert inner.calls == 1 and events[-1] == "run.completed"


async def test_a_different_question_is_a_miss() -> None:
    r, inner, _ = make()
    await run(r, text="how do i get a refund")
    await run(r, text="where is my shipment")
    assert inner.calls == 2


async def test_threshold_is_respected() -> None:
    r, inner, _ = make(threshold=0.9999)
    await run(r, text="how do i get a refund")
    await run(r, text="how can i get my money refunded")
    assert inner.calls == 2


async def test_hits_never_cross_users_agent_versions_or_tenants() -> None:
    r, inner, _ = make(scope="user")
    await run(r, text="how do i get a refund", user="alice")
    await run(r, text="how can i get my money refunded", user="bob")
    await run(r, text="how can i get my money refunded", user="alice", version="2")
    assert inner.calls == 3
    await run(r, text="how can i get my money refunded", user="alice")  # same user + version: hit
    assert inner.calls == 3


async def test_an_embedder_outage_degrades_to_normal_answering() -> None:
    r, inner, emb = make()
    await run(r, text="how do i get a refund")
    emb.fail = True
    events = await run(r, text="how can i get my money refunded")
    assert inner.calls == 2 and events[-1] == "run.completed"
