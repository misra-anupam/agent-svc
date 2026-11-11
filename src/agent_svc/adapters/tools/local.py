"""Local, dependency-free tools. Deliberately read-only and side-effect free (OWASP LLM06)."""

from __future__ import annotations

import ast
import operator
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from agent_svc.config.settings import Settings
from agent_svc.domain.exceptions import ToolExecutionError
from agent_svc.domain.models import ToolSpec

_BIN: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UN: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_MAX_EXPR = 200


def safe_eval(expression: str) -> float | int:
    """Arithmetic only: no names, calls or attributes, bounded exponent and size."""
    if len(expression) > _MAX_EXPR:
        raise ToolExecutionError("expression too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolExecutionError(f"invalid expression: {exc.msg}") from exc

    def ev(node: ast.AST) -> float | int:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return cast("float | int", node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ToolExecutionError("exponent too large")
            try:
                return _BIN[type(node.op)](left, right)  # type: ignore[no-any-return]
            except ZeroDivisionError as exc:
                raise ToolExecutionError("division by zero") from exc
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
            return _UN[type(node.op)](ev(node.operand))  # type: ignore[no-any-return]
        raise ToolExecutionError("only arithmetic expressions are allowed")

    return ev(tree)


@dataclass(frozen=True, slots=True)
class LocalTool:
    spec: ToolSpec
    fn: Callable[[dict[str, Any]], Awaitable[str]]


async def _calculator(args: dict[str, Any]) -> str:
    return str(safe_eval(str(args.get("expression", ""))))


async def _current_time(args: dict[str, Any]) -> str:
    tz = str(args.get("timezone") or "UTC")
    try:
        zone = ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ToolExecutionError(f"unknown timezone {tz!r}") from exc
    return datetime.now(UTC).astimezone(zone).isoformat(timespec="seconds")


BUILTIN: dict[str, LocalTool] = {
    "calculator": LocalTool(
        ToolSpec(
            "calculator",
            "Evaluate an arithmetic expression, e.g. '(2 + 3) * 4'.",
            {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        ),
        _calculator,
    ),
    "current_time": LocalTool(
        ToolSpec(
            "current_time",
            "Current date and time in an IANA timezone (default UTC).",
            {"type": "object", "properties": {"timezone": {"type": "string"}}},
        ),
        _current_time,
    ),
}


class LocalToolProvider:
    def __init__(self, tools: list[LocalTool]) -> None:
        self._tools = {t.spec.name: t for t in tools}

    async def list_tools(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolExecutionError(f"unknown tool {name!r}")
        return await tool.fn(arguments)


def build(settings: Settings) -> LocalToolProvider | None:
    names = settings.tools.local
    return LocalToolProvider([BUILTIN[n] for n in names]) if names else None
