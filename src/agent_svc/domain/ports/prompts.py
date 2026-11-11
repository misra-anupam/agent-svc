from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class PromptVersion:
    name: str
    version: str
    template: str


class PromptStore(Protocol):
    async def get(
        self, name: str, *, version: str | None = None, alias: str | None = None
    ) -> PromptVersion: ...
    async def register(self, name: str, template: str) -> PromptVersion: ...
