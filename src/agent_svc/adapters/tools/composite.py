"""Tool provider decorators: Composite (merge sources) and Guarded (policy + safety)."""

from __future__ import annotations

import asyncio
import fnmatch
from typing import Any

import jsonschema
import structlog

from agent_svc.domain.exceptions import DomainError, ToolExecutionError
from agent_svc.domain.models import ToolSpec
from agent_svc.domain.ports import ToolProvider

log = structlog.get_logger()


class CompositeToolProvider:
    """Merges several providers; the first provider to expose a name wins."""

    def __init__(self, providers: list[ToolProvider]) -> None:
        self._providers = providers
        self._owner: dict[str, ToolProvider] = {}

    async def list_tools(self) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        owner: dict[str, ToolProvider] = {}
        for p in self._providers:
            for spec in await p.list_tools():
                if spec.name in owner:
                    log.warning("duplicate_tool_ignored", tool=spec.name)
                    continue
                owner[spec.name] = p
                specs.append(spec)
        self._owner = owner
        return specs

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self._owner:
            await self.list_tools()
        provider = self._owner.get(name)
        if provider is None:
            raise ToolExecutionError(f"unknown tool {name!r}")
        return await provider.call(name, arguments)


class GuardedToolProvider:
    """Allowlist, argument validation, timeout and output cap around any provider."""

    def __init__(
        self,
        inner: ToolProvider,
        *,
        allowlist: list[str] | None,
        timeout_seconds: float,
        max_result_chars: int,
    ) -> None:
        self._inner, self._allow = inner, allowlist
        self._timeout, self._max = timeout_seconds, max_result_chars
        self._schemas: dict[str, dict[str, Any]] = {}

    def _permitted(self, name: str) -> bool:
        return self._allow is None or any(fnmatch.fnmatchcase(name, pat) for pat in self._allow)

    async def list_tools(self) -> list[ToolSpec]:
        specs = [s for s in await self._inner.list_tools() if self._permitted(s.name)]
        self._schemas = {s.name: s.input_schema for s in specs}
        return specs

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        if not self._permitted(name):
            raise ToolExecutionError(f"tool {name!r} is not permitted")
        if name not in self._schemas:
            await self.list_tools()
        schema = self._schemas.get(name)
        if schema:
            try:
                jsonschema.validate(arguments, schema)
            except jsonschema.ValidationError as exc:
                raise ToolExecutionError(f"invalid arguments for {name}: {exc.message}") from exc
        try:
            async with asyncio.timeout(self._timeout):
                result = await self._inner.call(name, arguments)
        except TimeoutError as exc:
            raise ToolExecutionError(f"tool {name!r} timed out after {self._timeout:g}s") from exc
        except DomainError:
            raise
        except Exception as exc:
            log.warning("tool_failed", tool=name, error=repr(exc))
            raise ToolExecutionError(f"tool {name!r} failed ({type(exc).__name__})") from exc
        return result if len(result) <= self._max else result[: self._max] + "…[truncated]"
