from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class PromptVersion:
    name: str
    version: str
    template: str

    @property
    def ref(self) -> str:
        return f"{self.name}:{self.version}"


class PromptStore(Protocol):
    """Versioned prompts with movable aliases (``prod``, ``staging``)."""

    async def get(
        self, name: str, *, version: str | None = None, alias: str | None = None
    ) -> PromptVersion:
        """Latest version when neither ``version`` nor ``alias`` is given."""
        ...

    async def register(self, name: str, template: str) -> PromptVersion: ...

    async def set_alias(self, name: str, alias: str, version: str) -> None: ...
