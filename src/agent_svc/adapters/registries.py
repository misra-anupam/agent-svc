"""Central registries, one per port. Built-ins are lazy "module:attr" strings."""

from __future__ import annotations

from agent_svc.adapters.llm.stack import build_llm_stack
from agent_svc.config.settings import Settings
from agent_svc.domain.ports import AgentRunner, LLMClient
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


def build_llm(settings: Settings) -> LLMClient:
    s = settings.llm
    return build_llm_stack(s, lambda model: llm_providers.create(s.provider, settings, model))


def build_runner(settings: Settings, llm: LLMClient) -> AgentRunner:
    return orchestrators.create(settings.orchestrator.provider, settings, llm)
