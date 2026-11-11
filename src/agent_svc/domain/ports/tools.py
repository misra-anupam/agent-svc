from __future__ import annotations

from typing import Any, Protocol

from agent_svc.domain.models import ToolSpec


class ToolProvider(Protocol):
    """Source of tools (local registry, MCP servers, ...)."""

    async def list_tools(self) -> list[ToolSpec]: ...
    async def call(self, name: str, arguments: dict[str, Any]) -> Any: ...
