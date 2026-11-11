"""Central registries, one per port. Built-ins are lazy "module:attr" strings."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent_svc.adapters import factories as fx
from agent_svc.adapters.llm.stack import build_llm_stack
from agent_svc.config.settings import Settings
from agent_svc.domain.models import Principal
from agent_svc.domain.ports import (
    AgentRunner,
    Authenticator,
    Cache,
    Embedder,
    Evaluator,
    Guardrail,
    LLMClient,
    LongTermMemory,
    PromptStore,
    RunBuffer,
    ThreadMemory,
    ToolProvider,
    Tracer,
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


# Guardrails: factory(settings) -> Guardrail
guardrails: Registry[Guardrail] = Registry("guardrails")
guardrails.register("builtin", "agent_svc.adapters.guardrails.builtin:build")
guardrails.register("presidio", "agent_svc.adapters.guardrails.presidio:build", extra="presidio")
guardrails.register(
    "guardrails_ai", "agent_svc.adapters.guardrails.guardrails_ai:build", extra="guardrails_ai"
)
# no uv extra: llm-guard's pins conflict with the shared lock (archived upstream); see adapter docs
guardrails.register("llm_guard", "agent_svc.adapters.guardrails.llm_guard:build")
guardrails.register("nemo", "agent_svc.adapters.guardrails.http_rails:build_nemo")
guardrails.register("llama_guard", "agent_svc.adapters.guardrails.http_rails:build_llama_guard")
guardrails.register("prompt_guard", "agent_svc.adapters.guardrails.http_rails:build_prompt_guard")


# Prompt stores: factory(settings) -> PromptStore
prompt_stores: Registry[PromptStore] = Registry("prompts")
prompt_stores.register("file", "agent_svc.adapters.prompts.file:build")
prompt_stores.register("mlflow", "agent_svc.adapters.prompts.mlflow:build", extra="mlflow_genai")
prompt_stores.register("langsmith", "agent_svc.adapters.prompts.langsmith:build", extra="langsmith")

# Evaluators: factory(settings) -> Evaluator
evaluators: Registry[Evaluator] = Registry("eval")
evaluators.register("local", "agent_svc.adapters.eval.local:build")
evaluators.register(
    "mlflow_genai", "agent_svc.adapters.eval.mlflow_genai:build", extra="mlflow_genai"
)
evaluators.register("langsmith", "agent_svc.adapters.eval.langsmith:build", extra="langsmith")
evaluators.register("adk_eval", "agent_svc.adapters.eval.adk_eval:build", extra="adk_eval")


def build_prompt_store(settings: Settings) -> PromptStore:
    from agent_svc.adapters.prompts.cached import CachedPromptStore

    return CachedPromptStore(
        prompt_stores.create(settings.prompts.provider, settings),
        settings.prompts.cache_ttl_seconds,
    )


# Tool sources: factory(settings) -> ToolProvider | None (None = not configured)
tool_sources: Registry[ToolProvider | None] = Registry("tools")
tool_sources.register("local", "agent_svc.adapters.tools.local:build")
tool_sources.register("mcp", "agent_svc.adapters.tools.mcp:build", extra="mcp")

tracers: Registry[Tracer] = Registry("tracing")
tracers.register("none", "agent_svc.adapters.tracing.noop:build")
tracers.register("otel", "agent_svc.adapters.tracing.otel:build", extra="otel")
tracers.register("mlflow", "agent_svc.adapters.tracing.mlflow:build", extra="mlflow")
tracers.register("langfuse", "agent_svc.adapters.tracing.langfuse:build", extra="langfuse")

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


def build_tool_sources(settings: Settings) -> list[ToolProvider]:
    sources = (tool_sources.create(name, settings) for name in tool_sources.names())
    return [s for s in sources if s is not None]


def build_guardrail_policy(settings: Settings) -> Callable[[Principal], Any]:
    from agent_svc.adapters.guardrails.policy import TierPolicyResolver

    return TierPolicyResolver(settings.guardrails, lambda n: guardrails.create(n, settings))


def build_runner(settings: Settings, llm: LLMClient, tools: ToolProvider) -> AgentRunner:
    return orchestrators.create(settings.orchestrator.provider, settings, llm, tools)
