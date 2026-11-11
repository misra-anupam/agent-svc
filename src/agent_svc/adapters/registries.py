"""Central registries, one per port. Built-ins are lazy "module:attr" strings."""

from __future__ import annotations

from agent_svc.config.settings import Settings
from agent_svc.domain.ports import AgentRunner
from agent_svc.registry import Registry

orchestrators: Registry[AgentRunner] = Registry("orchestrators")
orchestrators.register("echo", "agent_svc.adapters.orchestrators.echo:build_echo")
# Registered in later phases (each needs its optional extra):
orchestrators.register(
    "langgraph", "agent_svc.adapters.orchestrators.langgraph:build", extra="langgraph"
)
orchestrators.register("adk", "agent_svc.adapters.orchestrators.adk:build", extra="adk")
orchestrators.register(
    "deepagents", "agent_svc.adapters.orchestrators.deepagents:build", extra="deepagents"
)
orchestrators.register("maf", "agent_svc.adapters.orchestrators.maf:build", extra="maf")
orchestrators.register("crewai", "agent_svc.adapters.orchestrators.crewai:build", extra="crewai")


def build_runner(settings: Settings) -> AgentRunner:
    return orchestrators.create(settings.orchestrator.provider, settings)
