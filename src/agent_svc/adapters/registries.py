"""Central registries, one per port. Built-ins are lazy "module:attr" strings."""

from __future__ import annotations

from agent_svc.adapters import factories as fx
from agent_svc.adapters.llm.stack import build_llm_stack
from agent_svc.config.settings import Settings
from agent_svc.domain.ports import (
    AgentRunner,
    Authenticator,
    Cache,
    Embedder,
    LLMClient,
    LongTermMemory,
    RunBuffer,
    ThreadMemory,
    UserMemory,
)
from agent_svc.registry import Registry

# Base (unwrapped) LLM providers: factory(settings, model_name) -> LLMClient
llm_providers: Registry[LLMClient] = Registry("llm")
llm_providers.register("litellm", "agent_svc.adapters.llm.litellm_client:build", extra="litellm")
llm_providers.register("fake", "agent_svc.adapters.llm.fake:build_fake")

# Orchestrators: factory(settings, llm) -> AgentRunner
orchestrators: Registry[AgentRunner] = Registry("orchestrators")
orchestrators.register("echo", "agent_svc.adapters.orchestrators.echo:build_echo")
orchestrators.register(
    "langgraph", "agent_svc.adapters.orchestrators.langgraph:build", extra="langgraph"
)
orchestrators.register("adk", "agent_svc.adapters.orchestrators.adk:build", extra="adk")
orchestrators.register(
    "deepagents", "agent_svc.adapters.orchestrators.deepagents:build", extra="deepagents"
)
orchestrators.register("maf", "agent_svc.adapters.orchestrators.maf:build", extra="maf")
orchestrators.register("crewai", "agent_svc.adapters.orchestrators.crewai:build", extra="crewai")


# --- state, protection, auth: factories take (settings, infra) ---------------------------------
authenticators: Registry[Authenticator] = Registry("auth")
authenticators.register("noauth", "agent_svc.adapters.auth.noauth:build")
authenticators.register("jwt", "agent_svc.adapters.auth.jwt:build")

embedders: Registry[Embedder] = Registry("embedding")
embedders.register("fake", "agent_svc.adapters.embedding.embedders:build_fake")
embedders.register(
    "litellm", "agent_svc.adapters.embedding.embedders:build_litellm", extra="litellm"
)

limit_backends: Registry[fx.LimitBackend] = Registry("ratelimit")
limit_backends.register("inmemory", fx.limits_inmemory)
limit_backends.register("redis", fx.limits_redis)

caches: Registry[Cache] = Registry("cache")
caches.register("inmemory", fx.cache_inmemory)
caches.register("redis", fx.cache_redis)

thread_memories: Registry[ThreadMemory] = Registry("thread_memory")
thread_memories.register("inmemory", fx.thread_inmemory)
thread_memories.register("redis", fx.thread_redis)

user_memories: Registry[UserMemory] = Registry("user_memory")
user_memories.register("inmemory", fx.user_inmemory)
user_memories.register("postgres", fx.user_postgres)

long_term_memories: Registry[LongTermMemory] = Registry("long_term_memory")
long_term_memories.register("inmemory", fx.long_term_inmemory)
long_term_memories.register("pgvector", fx.long_term_pgvector)

run_buffers: Registry[RunBuffer] = Registry("run_buffer")
run_buffers.register("inmemory", fx.buffer_inmemory)
run_buffers.register("redis", fx.buffer_redis)


def build_llm(settings: Settings) -> LLMClient:
    s = settings.llm
    return build_llm_stack(s, lambda model: llm_providers.create(s.provider, settings, model))


def build_runner(settings: Settings, llm: LLMClient) -> AgentRunner:
    return orchestrators.create(settings.orchestrator.provider, settings, llm)
