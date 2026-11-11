"""Needs Docker (testcontainers + pgvector image). Skipped automatically when unavailable."""

import asyncpg
import pytest

from agent_svc.adapters.embedding.embedders import FakeEmbedder
from agent_svc.adapters.infra.schema import ensure_schema
from agent_svc.adapters.memory.long_term import PgVectorLongTermMemory
from agent_svc.adapters.memory.user import PostgresUserMemory

pytestmark = pytest.mark.integration
DIM = 64


@pytest.fixture(scope="module")
def pg_dsn():  # type: ignore[no-untyped-def]
    try:
        from testcontainers.community.postgres import PostgresContainer

        container = PostgresContainer("pgvector/pgvector:pg16", driver=None)
        container.start()
    except Exception as exc:  # docker missing / image pull failed
        pytest.skip(f"docker unavailable: {exc}")
    yield container.get_connection_url()
    container.stop()


@pytest.fixture
async def pool(pg_dsn: str):  # type: ignore[no-untyped-def]
    p = await asyncpg.create_pool(pg_dsn, min_size=1, max_size=3)
    await ensure_schema(p, embedding_dim=DIM, vector=True)
    await ensure_schema(p, embedding_dim=DIM, vector=True)  # idempotent
    yield p
    await p.execute("TRUNCATE user_memory, long_term_memory")
    await p.close()


async def test_user_memory_upsert_merge_delete(pool: asyncpg.Pool) -> None:
    m = PostgresUserMemory(pool)
    await m.merge("t", "alice", {"lang": "en", "n": 1})
    await m.merge("t", "alice", {"n": 2, "tz": "UTC"})
    assert await m.get("t", "alice") == {"lang": "en", "n": 2, "tz": "UTC"}
    assert await m.get("t", "bob") == {}
    await m.delete("t", "alice")
    assert await m.get("t", "alice") == {}


async def test_pgvector_recall_is_ranked_and_namespaced(pool: asyncpg.Pool) -> None:
    lt = PgVectorLongTermMemory(pool, FakeEmbedder(DIM))
    await lt.remember("t", "alice", "my favourite colour is green", thread="a")
    await lt.remember("t", "alice", "kubernetes deploy pipeline notes")
    await lt.remember("t", "bob", "green is bobs colour too")
    hits = await lt.recall("t", "alice", "which colour green do I like", k=5)
    assert hits[0].text == "my favourite colour is green" and hits[0].metadata == {"thread": "a"}
    assert {h.text for h in hits} == {
        "my favourite colour is green",
        "kubernetes deploy pipeline notes",
    }
    await lt.forget("t", "alice")
    assert await lt.recall("t", "alice", "green") == []
    assert len(await lt.recall("t", "bob", "green")) == 1  # other user untouched
