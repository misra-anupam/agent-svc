"""deepagents adapter: planning, virtual filesystem and sub-agents (LangGraph under the hood).

The sub-agent tool (``task``) is surfaced as an ``agent`` step, so the UI folds a delegated
sub-agent's own tool calls under it.
"""

from __future__ import annotations

from typing import Any

from deepagents import create_deep_agent
from langchain_core.tools import BaseTool

from agent_svc.adapters.orchestrators.langchain_bridge import LangGraphRunner, PortChatModel
from agent_svc.config.settings import Settings
from agent_svc.domain.models import StepKind
from agent_svc.domain.ports import LLMClient, ToolProvider


def build(settings: Settings, llm: LLMClient, tools: ToolProvider) -> LangGraphRunner:
    model = PortChatModel(llm=llm)

    async def build_graph(lc_tools: list[BaseTool]) -> Any:
        return create_deep_agent(
            model=model, tools=lc_tools, system_prompt=settings.orchestrator.system_prompt
        )

    return LangGraphRunner(
        build_graph,
        tools,
        max_iterations=settings.orchestrator.max_iterations,
        kind_overrides={"task": StepKind.AGENT},
    )
