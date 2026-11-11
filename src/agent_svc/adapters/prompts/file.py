"""Filesystem prompt store: ``<root>/<name>/<n>.txt`` plus ``<root>/<name>/aliases.json``.

Plain files, so prompts live in git and review like code. Names and aliases are validated
(no path traversal); writes are atomic.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

import anyio

from agent_svc.config.settings import Settings
from agent_svc.domain.exceptions import BadRequestError, NotFoundError
from agent_svc.domain.ports import PromptVersion

_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _check(kind: str, value: str) -> str:
    if not _SAFE.match(value):
        raise BadRequestError(f"invalid prompt {kind} {value!r}")
    return value


class FilePromptStore:
    name = "file"

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def _dir(self, name: str) -> Path:
        return self._root / _check("name", name)

    @staticmethod
    def _versions(d: Path) -> list[int]:
        return sorted(int(p.stem) for p in d.glob("*.txt") if p.stem.isdigit())

    def _aliases(self, d: Path) -> dict[str, str]:
        f = d / "aliases.json"
        return json.loads(f.read_text()) if f.exists() else {}

    def _get(self, name: str, version: str | None, alias: str | None) -> PromptVersion:
        d = self._dir(name)
        versions = self._versions(d) if d.is_dir() else []
        if not versions:
            raise NotFoundError(f"prompt {name!r} not found")
        if alias is not None:
            target = self._aliases(d).get(_check("alias", alias))
            if target is None:
                raise NotFoundError(f"prompt {name!r} has no alias {alias!r}")
            version = target
        v = int(_check("version", version)) if version is not None else versions[-1]
        f = d / f"{v}.txt"
        if not f.is_file():
            raise NotFoundError(f"prompt {name!r} has no version {v}")
        return PromptVersion(name, str(v), f.read_text())

    def _atomic_write(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent)
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)

    def _register(self, name: str, template: str) -> PromptVersion:
        d = self._dir(name)
        versions = self._versions(d) if d.is_dir() else []
        v = (versions[-1] if versions else 0) + 1
        self._atomic_write(d / f"{v}.txt", template)
        return PromptVersion(name, str(v), template)

    def _set_alias(self, name: str, alias: str, version: str) -> None:
        d = self._dir(name)
        self._get(name, version, None)  # must exist
        aliases = self._aliases(d)
        aliases[_check("alias", alias)] = str(int(version))
        self._atomic_write(d / "aliases.json", json.dumps(aliases, indent=2, sort_keys=True))

    async def get(
        self, name: str, *, version: str | None = None, alias: str | None = None
    ) -> PromptVersion:
        return await anyio.to_thread.run_sync(self._get, name, version, alias)

    async def register(self, name: str, template: str) -> PromptVersion:
        return await anyio.to_thread.run_sync(self._register, name, template)

    async def set_alias(self, name: str, alias: str, version: str) -> None:
        await anyio.to_thread.run_sync(self._set_alias, name, alias, version)


def build(settings: Settings) -> FilePromptStore:
    return FilePromptStore(settings.prompts.path)
