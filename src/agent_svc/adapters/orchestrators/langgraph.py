"""LangGraph adapter: a tool-calling ReAct agent (``langchain.agents.create_agent``)."""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import BaseTool

from agent_svc.adapters.orchestrators.langchain_bridge import LangGraphRunner, PortChatModel
from agent_svc.config.settings import Settings
from agent_svc.domain.ports import LLMClient, ToolProvider


def build(settings: Settings, llm: LLMClient, tools: ToolProvider) -> LangGraphRunner:
    model = PortChatModel(llm=llm)

    async def build_graph(lc_tools: list[BaseTool]) -> Any:
        return create_agent(
            model, lc_tools, system_prompt=settings.orchestrator.system_prompt, name="agent"
        )

    return LangGraphRunner(build_graph, tools, max_iterations=settings.orchestrator.max_iterations)
