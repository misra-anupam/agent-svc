from __future__ import annotations

from pathlib import Path

import pytest

from agent_svc.adapters.prompts.cached import CachedPromptStore
from agent_svc.adapters.prompts.mlflow import MlflowPromptStore
from agent_svc.container import parse_prompt_ref
from agent_svc.domain.exceptions import NotFoundError
from agent_svc.domain.ports import PromptVersion


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MlflowPromptStore:
    """MLflow prompt registry on a throwaway sqlite file (offline, no server)."""
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    return MlflowPromptStore(f"sqlite:///{tmp_path}/m.db")


async def test_versions_increment_and_latest_is_default(store: MlflowPromptStore) -> None:
    assert (await store.register("system", "v one")).version == "1"
    assert (await store.register("system", "v two")).version == "2"
    assert (await store.get("system")).template == "v two"
    assert (await store.get("system", version="1")).template == "v one"


async def test_aliases_move_between_versions(store: MlflowPromptStore) -> None:
    await store.register("system", "a")
    await store.register("system", "b")
    await store.set_alias("system", "prod", "1")
    assert (await store.get("system", alias="prod")).template == "a"
    await store.set_alias("system", "prod", "2")
    assert (await store.get("system", alias="prod")).template == "b"
    with pytest.raises(NotFoundError):
        await store.set_alias("system", "prod", "9")
    with pytest.raises(NotFoundError):
        await store.get("system", alias="staging")


async def test_missing_prompt_is_not_found(store: MlflowPromptStore) -> None:
    with pytest.raises(NotFoundError):
        await store.get("nope")


class Counting:
    def __init__(self) -> None:
        self.gets = 0

    async def get(self, name: str, *, version: str | None = None, alias: str | None = None):  # type: ignore[no-untyped-def]
        self.gets += 1
        return PromptVersion(name, version or str(self.gets), f"t{self.gets}")

    async def register(self, name: str, template: str) -> PromptVersion:
        return PromptVersion(name, "1", template)

    async def set_alias(self, name: str, alias: str, version: str) -> None:
        return None


async def test_cache_expires_moving_refs_but_pins_versions_forever() -> None:
    now = [0.0]
    inner = Counting()
    c = CachedPromptStore(inner, 60, clock=lambda: now[0])
    await c.get("p", alias="prod")
    await c.get("p", alias="prod")
    assert inner.gets == 1
    now[0] = 61
    await c.get("p", alias="prod")
    assert inner.gets == 2  # alias entry expired
    await c.get("p", version="3")
    now[0] = 10_000
    await c.get("p", version="3")
    assert inner.gets == 3  # a pinned version never expires


async def test_register_invalidates_that_prompt() -> None:
    inner = Counting()
    c = CachedPromptStore(inner, 60)
    await c.get("p", alias="prod")
    await c.register("p", "new")
    await c.get("p", alias="prod")
    assert inner.gets == 2


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("sys", ("sys", None, None)),
        ("sys:3", ("sys", "3", None)),
        ("sys@prod", ("sys", None, "prod")),
    ],
)
def test_prompt_ref_parsing(ref: str, expected: tuple[str, str | None, str | None]) -> None:
    assert parse_prompt_ref(ref) == expected
