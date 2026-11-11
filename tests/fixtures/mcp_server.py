"""Tiny MCP server for tests and the `mcp-demo` compose profile.

python mcp_server.py stdio
python mcp_server.py streamable-http 8080
"""

import sys

from mcp.server.fastmcp import FastMCP

port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
mcp = FastMCP("demo", host="0.0.0.0", port=port)


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@mcp.tool()
def boom() -> str:
    """Always fails."""
    raise ValueError("kaboom")


@mcp.tool()
def big() -> str:
    """Returns a very long string."""
    return "x" * 50_000


if __name__ == "__main__":
    mcp.run(transport=sys.argv[1] if len(sys.argv) > 1 else "stdio")  # type: ignore[arg-type]
