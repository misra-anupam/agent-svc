import fakeredis
import pytest

from agent_svc.adapters.embedding.embedders import FakeEmbedder
from agent_svc.adapters.memory.long_term import InMemoryLongTermMemory
from agent_svc.adapters.memory.thread import InMemoryThreadMemory, RedisThreadMemory
from agent_svc.adapters.memory.user import InMemoryUserMemory
from agent_svc.application.memory_facade import MemoryFacade
from agent_svc.config.settings import MemorySettings
from agent_svc.domain.models import Message, Principal


@pytest.fixture(params=["inmemory", "redis"])
def thread_mem(request: pytest.FixtureRequest):  # type: ignore[no-untyped-def]
    if request.param == "inmemory":
        return InMemoryThreadMemory(max_messages=5)
    return RedisThreadMemory(fakeredis.FakeAsyncRedis(), ttl_seconds=60, max_messages=5)


async def test_thread_memory_order_limit_and_cap(thread_mem) -> None:  # type: ignore[no-untyped-def]
    await thread_mem.append("t", "th", [Message("user", f"m{i}") for i in range(8)])
    got = await thread_mem.load("t", "th")
    assert [m.content for m in got] == ["m3", "m4", "m5", "m6", "m7"]  # capped at 5
    assert [m.content for m in await thread_mem.load("t", "th", limit=2)] == ["m6", "m7"]
    assert await thread_mem.load("t", "other") == []
    assert await thread_mem.load("other-tenant", "th") == []  # tenant isolation
    await thread_mem.delete("t", "th")
    assert await thread_mem.load("t", "th") == []


async def test_user_memory_merge_and_delete() -> None:
    u = InMemoryUserMemory()
    await u.merge("t", "alice", {"lang": "en"})
    await u.merge("t", "alice", {"tz": "UTC"})
    assert await u.get("t", "alice") == {"lang": "en", "tz": "UTC"}
    assert await u.get("t", "bob") == {}
    await u.delete("t", "alice")
    assert await u.get("t", "alice") == {}


async def test_long_term_recall_ranks_by_similarity_and_is_namespaced() -> None:
    lt = InMemoryLongTermMemory(FakeEmbedder(128))
    await lt.remember("t", "alice", "my favourite colour is green")
    await lt.remember("t", "alice", "the deploy runs on kubernetes")
    await lt.remember("t", "bob", "bob likes green tea a lot")
    hits = await lt.recall("t", "alice", "what colour do I like green", k=2)
    assert hits[0].text == "my favourite colour is green" and hits[0].score > hits[1].score
    assert all("bob" not in h.text for h in hits)  # never crosses users
    await lt.forget("t", "alice")
    assert await lt.recall("t", "alice", "green") == []


def _facade(max_tokens: int = 8000, **kw: object) -> MemoryFacade:
    return MemoryFacade(MemorySettings(max_context_tokens=max_tokens), **kw)  # type: ignore[arg-type]


P = Principal(subject="alice", tenant="t")


async def test_context_stitches_history_user_facts_and_long_term() -> None:
    user, lt = InMemoryUserMemory(), InMemoryLongTermMemory(FakeEmbedder(128))
    await user.merge("t", "alice", {"name": "Alice"})
    await lt.remember("t", "alice", "I like green tea")
    f = _facade(thread=InMemoryThreadMemory(), user=user, long_term=lt)
    await f.save_turn(P, "th", [Message("user", "hello")], "hi Alice")
    ctx = await f.build_context(P, "th", [Message("user", "what tea green do I like")])
    assert (
        ctx[0].role == "system" and "name=Alice" in ctx[0].content and "green tea" in ctx[0].content
    )
    assert [m.content for m in ctx[1:]] == ["hello", "hi Alice", "what tea green do I like"]


async def test_context_trims_oldest_history_first_and_keeps_new_messages() -> None:
    f = _facade(max_tokens=60, thread=InMemoryThreadMemory())
    old = [Message("user", "x" * 100 + str(i)) for i in range(6)]  # ~29 tokens each
    await f.thread.append("t", f.thread_key(P, "th"), old)  # type: ignore[union-attr]
    ctx = await f.build_context(P, "th", [Message("user", "new question")])
    assert ctx[-1].content == "new question"
    kept = [m.content for m in ctx[:-1]]
    assert 0 < len(kept) < 6 and kept == [m.content for m in old][-len(kept) :]


async def test_threads_are_private_to_their_owner() -> None:
    f = _facade(thread=InMemoryThreadMemory())
    mallory = Principal(subject="mallory", tenant="t")
    await f.save_turn(P, "shared-id", [Message("user", "secret plan")], "ok")
    ctx = await f.build_context(mallory, "shared-id", [Message("user", "hi")])
    assert [m.content for m in ctx] == ["hi"]  # same thread id, different owner: no history


async def test_memory_backend_failure_degrades_gracefully() -> None:
    class Broken:
        async def load(self, *a: object, **k: object) -> list[Message]:
            raise ConnectionError("down")

        async def append(self, *a: object, **k: object) -> None:
            raise ConnectionError("down")

    f = _facade(thread=Broken())  # type: ignore[arg-type]
    assert [m.content for m in await f.build_context(P, "t", [Message("user", "hi")])] == ["hi"]
    await f.save_turn(P, "t", [Message("user", "hi")], "yo")  # must not raise
