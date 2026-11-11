import asyncio
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from agent_svc.adapters.tools.composite import CompositeToolProvider, GuardedToolProvider
from agent_svc.adapters.tools.local import BUILTIN, LocalToolProvider
from agent_svc.adapters.tools.mcp import McpToolProvider, assert_url_allowed, build
from agent_svc.config import Settings
from agent_svc.config.settings import McpServerConfig, McpSettings
from agent_svc.domain.exceptions import ConfigurationError, ToolExecutionError

SERVER = str(Path(__file__).parent.parent / "fixtures" / "mcp_server.py")
STDIO = McpServerConfig(name="demo", transport="stdio", command=[sys.executable, SERVER, "stdio"])


@pytest.fixture
async def stdio_provider():  # type: ignore[no-untyped-def]
    p = McpToolProvider(McpSettings(servers=[STDIO], connect_timeout_seconds=30))
    await p.start()
    yield p
    await p.stop()


async def test_stdio_lists_namespaced_tools_with_mcp_source(stdio_provider) -> None:  # type: ignore[no-untyped-def]
    specs = {s.name: s for s in await stdio_provider.list_tools()}
    assert {"demo__add", "demo__boom", "demo__big"} <= set(specs)
    assert specs["demo__add"].source == "mcp:demo"
    assert specs["demo__add"].input_schema["properties"]["a"]["type"] == "integer"
    assert await stdio_provider.ping() is True


async def test_stdio_call_and_error_mapping(stdio_provider) -> None:  # type: ignore[no-untyped-def]
    assert await stdio_provider.call("demo__add", {"a": 2, "b": 3}) == "5"
    with pytest.raises(ToolExecutionError, match="kaboom"):
        await stdio_provider.call("demo__boom", {})
    with pytest.raises(ToolExecutionError, match="unknown MCP tool"):
        await stdio_provider.call("demo__nope", {})


async def test_guard_policy_over_mcp(stdio_provider) -> None:  # type: ignore[no-untyped-def]
    guarded = GuardedToolProvider(
        stdio_provider,
        allowlist=["demo__add", "demo__big"],
        timeout_seconds=10,
        max_result_chars=100,
    )
    assert {s.name for s in await guarded.list_tools()} == {"demo__add", "demo__big"}
    with pytest.raises(ToolExecutionError, match="not permitted"):
        await guarded.call("demo__boom", {})
    with pytest.raises(ToolExecutionError, match="invalid arguments"):
        await guarded.call("demo__add", {"a": "x", "b": 1})  # schema says integer
    big = await guarded.call("demo__big", {})
    assert len(big) < 120 and big.endswith("[truncated]")


async def test_composite_merges_local_and_mcp_without_shadowing(stdio_provider) -> None:  # type: ignore[no-untyped-def]
    comp = CompositeToolProvider([LocalToolProvider([BUILTIN["calculator"]]), stdio_provider])
    names = {s.name for s in await comp.list_tools()}
    assert {"calculator", "demo__add"} <= names
    assert await comp.call("calculator", {"expression": "6*7"}) == "42"
    assert await comp.call("demo__add", {"a": 1, "b": 1}) == "2"


async def test_guard_times_out_slow_tools() -> None:
    class Slow:
        async def list_tools(self):  # type: ignore[no-untyped-def]
            return []

        async def call(self, name: str, arguments: dict) -> str:  # type: ignore[type-arg]
            await asyncio.sleep(5)
            return "late"

    g = GuardedToolProvider(Slow(), allowlist=None, timeout_seconds=0.1, max_result_chars=10)
    with pytest.raises(ToolExecutionError, match="timed out"):
        await g.call("slow", {})


@pytest.mark.parametrize(
    ("url", "msg"),
    [
        ("http://169.254.169.254/latest/meta-data", "metadata"),
        ("http://metadata.google.internal/x", "metadata"),
        ("http://evil.example.com/mcp", "SSRF guard"),
        ("http://127.0.0.1:9/mcp", "SSRF guard"),  # loopback is NOT implicitly trusted
        ("file:///etc/passwd", "http"),
        ("gopher://x/", "http"),
        ("http:///nohost", "http"),
    ],
)
def test_ssrf_guard_blocks(url: str, msg: str) -> None:
    with pytest.raises(ConfigurationError, match=msg):
        assert_url_allowed(url, ["mcp-demo"])


def test_ssrf_guard_allows_listed_host_but_never_metadata() -> None:
    assert_url_allowed("http://mcp-demo:8080/mcp", ["mcp-demo"])
    assert_url_allowed("https://MCP-Demo/mcp", ["mcp-demo"])  # case-insensitive
    with pytest.raises(ConfigurationError, match="metadata"):
        assert_url_allowed("http://169.254.169.254/", ["169.254.169.254"])  # even if allowlisted


def test_build_fails_fast_on_disallowed_url_and_noops_without_servers() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert build(s) is None
    s.mcp.servers = [McpServerConfig(name="x", transport="streamable_http", url="http://evil/mcp")]
    with pytest.raises(ConfigurationError, match="SSRF"):
        build(s)


async def test_unreachable_server_is_skipped_unless_required() -> None:
    dead = McpServerConfig(name="dead", transport="stdio", command=["/nonexistent/binary"])
    soft = McpToolProvider(McpSettings(servers=[dead, STDIO], connect_timeout_seconds=30))
    await soft.start()
    try:
        assert {s.name.split("__")[0] for s in await soft.list_tools()} == {"demo"}
    finally:
        await soft.stop()
    hard = McpToolProvider(McpSettings(servers=[dead], connect_timeout_seconds=5, required=True))
    with pytest.raises(Exception):  # noqa: B017 - any connection failure must propagate
        await hard.start()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def http_server() -> Iterator[int]:
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, SERVER, "streamable-http", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            try:
                httpx.get(f"http://127.0.0.1:{port}/", timeout=0.5)
                break
            except httpx.TransportError:
                time.sleep(0.1)
        yield port
    finally:
        proc.terminate()
        proc.wait(5)


async def test_streamable_http_server_over_allowlisted_host(http_server: int) -> None:
    cfg = McpSettings(
        servers=[
            McpServerConfig(
                name="web", transport="streamable_http", url=f"http://127.0.0.1:{http_server}/mcp"
            )
        ],
        allowed_hosts=["127.0.0.1"],
        connect_timeout_seconds=20,
    )
    p = McpToolProvider(cfg)
    await p.start()
    try:
        assert "web__add" in {s.name for s in await p.list_tools()}
        assert await p.call("web__add", {"a": 20, "b": 22}) == "42"
    finally:
        await p.stop()


async def test_http_server_not_on_allowlist_is_never_contacted(http_server: int) -> None:
    cfg = McpSettings(
        servers=[
            McpServerConfig(
                name="web", transport="streamable_http", url=f"http://127.0.0.1:{http_server}/mcp"
            )
        ],
        allowed_hosts=[],
        required=True,
    )
    with pytest.raises(ConfigurationError, match="SSRF"):
        await McpToolProvider(cfg).start()
