"""Factory functions registered in adapters/registries.py: (settings, infra, ...)."""

from __future__ import annotations

from dataclasses import dataclass

from agent_svc.adapters.cache.stores import RedisCache
from agent_svc.adapters.infra import Infra
from agent_svc.adapters.memory.long_term import InMemoryLongTermMemory, PgVectorLongTermMemory
from agent_svc.adapters.memory.thread import InMemoryThreadMemory, RedisThreadMemory
from agent_svc.adapters.memory.user import InMemoryUserMemory, PostgresUserMemory
from agent_svc.adapters.ratelimit.redis import (
    RedisConcurrencyGate,
    RedisRateLimiter,
    RedisTokenBudget,
)
from agent_svc.adapters.runs.buffer import InMemoryRunBuffer, RedisRunBuffer
from agent_svc.config.settings import Settings
from agent_svc.domain.exceptions import ConfigurationError
from agent_svc.domain.ports import (
    Cache,
    ConcurrencyGate,
    Embedder,
    LongTermMemory,
    RateLimiter,
    RunBuffer,
    ThreadMemory,
    TokenBudget,
    UserMemory,
)


def _need(value: object, what: str, setting: str) -> None:
    if value is None:
        raise ConfigurationError(f"{setting} requires {what} to be enabled")


@dataclass(frozen=True, slots=True)
class LimitBackend:
    limiter: RateLimiter
    budget: TokenBudget
    gate: ConcurrencyGate


def limits_redis(_s: Settings, i: Infra) -> LimitBackend:
    _need(i.redis, "redis", "AGENT_RATELIMIT__PROVIDER=redis")
    return LimitBackend(
        RedisRateLimiter(i.redis), RedisTokenBudget(i.redis), RedisConcurrencyGate(i.redis)
    )


def cache_redis(_s: Settings, i: Infra) -> Cache:
    _need(i.redis, "redis", "AGENT_CACHE__PROVIDER=redis")
    return RedisCache(i.redis)


def thread_inmemory(s: Settings, _i: Infra) -> ThreadMemory:
    return InMemoryThreadMemory(s.memory.max_thread_messages)


def thread_redis(s: Settings, i: Infra) -> ThreadMemory:
    _need(i.redis, "redis", "AGENT_MEMORY__THREAD_PROVIDER=redis")
    return RedisThreadMemory(
        i.redis, ttl_seconds=s.memory.thread_ttl_seconds, max_messages=s.memory.max_thread_messages
    )


def user_inmemory(_s: Settings, _i: Infra) -> UserMemory:
    return InMemoryUserMemory()


def user_postgres(_s: Settings, i: Infra) -> UserMemory:
    _need(i.pg_pool, "postgres", "AGENT_MEMORY__USER_PROVIDER=postgres")
    return PostgresUserMemory(i.pg_pool)


def long_term_inmemory(_s: Settings, _i: Infra, embedder: Embedder) -> LongTermMemory:
    return InMemoryLongTermMemory(embedder)


def long_term_pgvector(_s: Settings, i: Infra, embedder: Embedder) -> LongTermMemory:
    _need(i.pg_pool, "postgres", "AGENT_MEMORY__LONG_TERM_PROVIDER=pgvector")
    return PgVectorLongTermMemory(i.pg_pool, embedder)


def buffer_inmemory(s: Settings, _i: Infra) -> RunBuffer:
    return InMemoryRunBuffer(s.app.run_buffer_ttl_seconds)


def buffer_redis(s: Settings, i: Infra) -> RunBuffer:
    _need(i.redis, "redis", "AGENT_APP__RUN_BUFFER=redis")
    return RedisRunBuffer(i.redis, s.app.run_buffer_ttl_seconds)
