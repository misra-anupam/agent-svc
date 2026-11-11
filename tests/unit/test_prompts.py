from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_svc.adapters.prompts.cached import CachedPromptStore
from agent_svc.adapters.prompts.file import FilePromptStore
from agent_svc.adapters.prompts.langsmith import LangSmithPromptStore
from agent_svc.container import parse_prompt_ref
from agent_svc.domain.exceptions import BadRequestError, ConfigurationError, NotFoundError
from agent_svc.domain.ports import PromptVersion


@pytest.fixture
def store(tmp_path: Path) -> FilePromptStore:
    return FilePromptStore(tmp_path)


async def test_versions_increment_and_latest_is_default(store: FilePromptStore) -> None:
    assert (await store.register("system", "v one")).version == "1"
    assert (await store.register("system", "v two")).version == "2"
    assert (await store.get("system")).template == "v two"
    assert (await store.get("system", version="1")).template == "v one"


async def test_aliases_move_between_versions(store: FilePromptStore) -> None:
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


@pytest.mark.parametrize("bad", ["../etc/passwd", "a/b", "", ".hidden", "x" * 80, "a b"])
async def test_path_traversal_and_odd_names_are_rejected(store: FilePromptStore, bad: str) -> None:
    with pytest.raises(BadRequestError):
        await store.register(bad, "x")
    with pytest.raises(BadRequestError):
        await store.get(bad)


async def test_missing_prompt_is_not_found(store: FilePromptStore) -> None:
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


# ---- MLflow (offline: sqlite tracking store, no server) --------------------------------------
async def test_mlflow_prompt_registry_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    from agent_svc.adapters.prompts.mlflow import MlflowPromptStore

    s = MlflowPromptStore(f"sqlite:///{tmp_path}/m.db")
    assert (await s.register("system", "first")).version == "1"
    await s.register("system", "second")
    await s.set_alias("system", "prod", "1")
    assert (await s.get("system", alias="prod")).template == "first"
    assert (await s.get("system")).template == "second"
    assert (await s.get("system", version="2")).version == "2"
    with pytest.raises(NotFoundError):
        await s.get("missing")


# ---- LangSmith (stubbed client: the live service needs an account) ----------------------------
class StubClient:
    def __init__(self, template: object = "Be {tone}.") -> None:
        self.template, self.idents = template, []

    def pull_prompt_commit(self, ident: str) -> SimpleNamespace:
        self.idents.append(ident)
        return SimpleNamespace(commit_hash="abc123")

    def pull_prompt(self, ident: str) -> SimpleNamespace:
        return SimpleNamespace(template=self.template)


async def test_langsmith_maps_alias_to_tag_and_commit_hash_to_version() -> None:
    client = StubClient()
    pv = await LangSmithPromptStore(client).get("system", alias="prod")
    assert (
        client.idents == ["system:prod"] and pv.version == "abc123" and pv.template == "Be {tone}."
    )


async def test_langsmith_rejects_non_string_prompts_and_alias_writes() -> None:
    store = LangSmithPromptStore(StubClient(template=["chat", "messages"]))
    with pytest.raises(ConfigurationError, match="plain string"):
        await store.get("system")
    with pytest.raises(ConfigurationError, match="commit tags"):
        await store.set_alias("system", "prod", "abc")
