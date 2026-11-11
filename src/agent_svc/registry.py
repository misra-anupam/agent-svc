"""Generic plugin registry: name -> factory, lazily imported, extensible via entry points."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module, metadata
from typing import Any

from agent_svc.domain.exceptions import ConfigurationError

type Factory[T] = Callable[..., T] | str  # callable, or "package.module:attr" resolved lazily


class Registry[T]:
    """Maps provider names to factories for one port.

    * Built-in adapters are registered with lazy ``"module:attr"`` strings so optional SDKs
      are only imported when selected.
    * Third parties can plug in via the ``agent_svc.<group>`` entry-point group.
    """

    def __init__(self, port: str, *, entry_point_group: str | None = None) -> None:
        self.port = port
        self._factories: dict[str, tuple[Factory[T], str | None]] = {}
        self._group = entry_point_group or f"agent_svc.{port}"
        self._entry_points_loaded = False

    def register(self, name: str, factory: Factory[T], *, extra: str | None = None) -> None:
        """Register ``factory``; ``extra`` names the uv extra that provides its dependencies."""
        self._factories[name] = (factory, extra)

    def provider(
        self, name: str, *, extra: str | None = None
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def deco(fn: Callable[..., T]) -> Callable[..., T]:
            self.register(name, fn, extra=extra)
            return fn

        return deco

    def names(self) -> list[str]:
        self._load_entry_points()
        return sorted(self._factories)

    def create(self, name: str, *args: Any, **kwargs: Any) -> T:
        if name not in self._factories:
            self._load_entry_points()
        if name not in self._factories:
            available = ", ".join(self.names()) or "none"
            raise ConfigurationError(
                f"Unknown {self.port} provider {name!r}. Available: {available}"
            )
        factory, extra = self._factories[name]
        try:
            fn = self._resolve(factory)
            return fn(*args, **kwargs)
        except ImportError as exc:
            hint = f" Install the extra: `uv sync --extra {extra}`." if extra else ""
            raise ConfigurationError(
                f"{self.port} provider {name!r} could not be imported ({exc}).{hint}"
            ) from exc

    @staticmethod
    def _resolve(factory: Factory[T]) -> Callable[..., T]:
        if callable(factory):
            return factory
        module_name, _, attr = factory.partition(":")
        fn: Callable[..., T] = getattr(import_module(module_name), attr)
        return fn

    def _load_entry_points(self) -> None:
        if self._entry_points_loaded:
            return
        self._entry_points_loaded = True
        for ep in metadata.entry_points(group=self._group):
            self._factories.setdefault(ep.name, (ep.value, None))
