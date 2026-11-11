from __future__ import annotations

from typing import Any, Protocol

from agent_svc.domain.models import StepKind, ToolSpec


class ToolProvider(Protocol):
    """Source of tools (local functions, MCP servers, ...). ``call`` raises ToolExecutionError."""

    async def list_tools(self) -> list[ToolSpec]: ...
    async def call(self, name: str, arguments: dict[str, Any]) -> str: ...


def step_kind_for(spec: ToolSpec | None) -> StepKind:
    """Tools from MCP servers are shown as ``mcp`` steps, everything else as ``tool``."""
    return StepKind.MCP if spec is not None and spec.source.startswith("mcp") else StepKind.TOOL
