"""MCP adapter: connects to configured MCP servers (stdio or streamable HTTP) and exposes their
tools through the framework-neutral ToolProvider port.

Security (OWASP A10 SSRF / LLM03 supply chain / LLM06 excessive agency):
* servers come only from operator config, never from model or user input;
* HTTP servers must be on ``mcp.allowed_hosts``; redirects off the server's origin are never
  followed (mcp>=2 enforces it);
* cloud metadata endpoints are refused even when listed;
* tool names are namespaced ``<server>__<tool>`` so a server cannot shadow a local tool.
"""

from __future__ import annotations

import asyncio
import re
from contextlib import AsyncExitStack
from typing import Any
from urllib.parse import urlparse

import httpx2
import structlog

from agent_svc.config.settings import McpServerConfig, McpSettings, Settings
from agent_svc.domain.exceptions import ConfigurationError, ToolExecutionError
from agent_svc.domain.models import ToolSpec

log = structlog.get_logger()

_METADATA_HOSTS = {
    "169.254.169.254",
    "fd00:ec2::254",
    "metadata.google.internal",
    "100.100.100.200",
}
_SAFE = re.compile(r"[^A-Za-z0-9_-]")


def assert_url_allowed(url: str, allowed_hosts: list[str]) -> None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or not host:
        raise ConfigurationError(f"MCP url must be http(s) with a host: {url!r}")
    if host in _METADATA_HOSTS:
        raise ConfigurationError(f"MCP host {host!r} is a cloud metadata endpoint and is blocked")
    if host not in {h.lower() for h in allowed_hosts}:
        raise ConfigurationError(
            f"MCP host {host!r} is not in AGENT_MCP__ALLOWED_HOSTS (SSRF guard)"
        )


def _no_redirect_client(
    headers: dict[str, str] | None = None, timeout: Any = None, auth: Any = None
) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(
        follow_redirects=False,
        headers=headers,
        timeout=timeout or httpx2.Timeout(30.0, read=300.0),
        auth=auth,
    )


def _tool_name(server: str, tool: str) -> str:
    return f"{_SAFE.sub('_', server)}__{_SAFE.sub('_', tool)}"[:64]


class McpToolProvider:
    """Owns all MCP connections inside ONE background task.

    MCP transports rely on anyio cancel scopes, which must be exited by the task that entered
    them. Running every connect/close in a dedicated owner task makes ``start()``/``stop()``
    safe to call from any task (lifespan, tests, shutdown hooks).
    """

    name = "mcp"

    def __init__(self, cfg: McpSettings) -> None:
        self._cfg = cfg
        self.required = cfg.required
        self._sessions: dict[str, Any] = {}  # server -> ClientSession
        self._specs: dict[
            str, tuple[str, str, ToolSpec]
        ] = {}  # exposed name -> (server, tool, spec)
        self._read_timeout = float(max(cfg.connect_timeout_seconds, 30))
        self._task: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._error: BaseException | None = None

    async def start(self) -> None:
        self._ready, self._stop, self._error = asyncio.Event(), asyncio.Event(), None
        self._task = asyncio.create_task(self._run(), name="mcp-connections")
        await self._ready.wait()
        if self._error is not None:
            await self.stop()
            raise self._error

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        self._sessions.clear()
        self._specs.clear()

    async def _run(self) -> None:
        try:
            async with AsyncExitStack() as stack:
                for server in self._cfg.servers:
                    child = AsyncExitStack()
                    try:
                        async with asyncio.timeout(self._cfg.connect_timeout_seconds):
                            await self._connect(server, child)
                    except BaseException as exc:
                        await child.aclose()
                        if self._cfg.required or isinstance(exc, asyncio.CancelledError):
                            raise
                        log.warning(
                            "mcp_server_unavailable", server=server.name, error=repr(exc)[:200]
                        )
                        continue
                    stack.push_async_callback(child.aclose)
                await self._refresh()
                self._ready.set()
                await self._stop.wait()
        except BaseException as exc:
            self._error = exc
        finally:
            self._ready.set()

    async def _connect(self, server: McpServerConfig, stack: AsyncExitStack) -> None:
        from mcp import ClientSession

        if server.transport == "stdio":
            if not server.command:
                raise ConfigurationError(f"MCP server {server.name!r}: stdio needs a command")
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(command=server.command[0], args=server.command[1:])
            read, write = await stack.enter_async_context(stdio_client(params))
        else:
            if not server.url:
                raise ConfigurationError(f"MCP server {server.name!r}: http needs a url")
            assert_url_allowed(server.url, self._cfg.allowed_hosts)
            from mcp.client.streamable_http import streamable_http_client

            http = await stack.enter_async_context(_no_redirect_client())  # we own its lifetime
            read, write = await stack.enter_async_context(
                streamable_http_client(server.url, http_client=http)
            )
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._sessions[server.name] = session
        log.info("mcp_server_connected", server=server.name, transport=server.transport)

    async def _refresh(self) -> None:
        specs: dict[str, tuple[str, str, ToolSpec]] = {}
        for server, session in self._sessions.items():
            result = await session.list_tools()
            for t in result.tools:
                exposed = _tool_name(server, t.name)
                specs[exposed] = (
                    server,
                    t.name,
                    ToolSpec(
                        exposed,
                        t.description or t.name,
                        dict(t.input_schema or {"type": "object"}),
                        source=f"mcp:{server}",
                    ),
                )
        self._specs = specs

    async def ping(self) -> bool:
        try:
            for session in self._sessions.values():
                await asyncio.wait_for(session.send_ping(), 3)
            return True
        except Exception:
            return False

    async def list_tools(self) -> list[ToolSpec]:
        return [spec for _, _, spec in self._specs.values()]

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        entry = self._specs.get(name)
        if entry is None:
            raise ToolExecutionError(f"unknown MCP tool {name!r}")
        server, tool, _ = entry
        result = await self._sessions[server].call_tool(
            tool, arguments, read_timeout_seconds=self._read_timeout
        )
        from mcp.types import CallToolResult, TextContent

        if not isinstance(result, CallToolResult):  # e.g. the server asks for interactive input
            raise ToolExecutionError(f"MCP tool {name!r} needs interactive input (unsupported)")
        text = "\n".join(
            c.text if isinstance(c, TextContent) else f"[{getattr(c, 'type', 'content')}]"
            for c in result.content
        )
        if result.is_error:
            raise ToolExecutionError(text or f"MCP tool {name!r} reported an error")
        return text


def build(settings: Settings) -> McpToolProvider | None:
    if not settings.mcp.servers:
        return None
    for s in settings.mcp.servers:  # fail fast on bad config, before any connection
        if s.transport == "streamable_http" and s.url:
            assert_url_allowed(s.url, settings.mcp.allowed_hosts)
    return McpToolProvider(settings.mcp)
