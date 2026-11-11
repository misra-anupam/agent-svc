"""Idempotent schema bootstrap. Swap for Alembic migrations once the schema evolves."""

from __future__ import annotations

import asyncpg


async def ensure_schema(pool: asyncpg.Pool, *, embedding_dim: int, vector: bool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """CREATE TABLE IF NOT EXISTS user_memory (
                 tenant text NOT NULL, user_id text NOT NULL,
                 facts jsonb NOT NULL DEFAULT '{}'::jsonb,
                 updated_at timestamptz NOT NULL DEFAULT now(),
                 PRIMARY KEY (tenant, user_id))"""
        )
        if vector:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(
                f"""CREATE TABLE IF NOT EXISTS long_term_memory (
                      id bigserial PRIMARY KEY, tenant text NOT NULL, user_id text NOT NULL,
                      text text NOT NULL, meta jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                      embedding vector({int(embedding_dim)}) NOT NULL,
                      created_at timestamptz NOT NULL DEFAULT now())"""
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS long_term_memory_owner "
                "ON long_term_memory (tenant, user_id)"
            )
