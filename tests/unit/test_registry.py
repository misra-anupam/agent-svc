import pytest

from agent_svc.config import Settings
from agent_svc.domain.exceptions import ConfigurationError
from agent_svc.registry import Registry


def test_register_and_create() -> None:
    reg: Registry[str] = Registry("demo")
    reg.register("a", lambda x: f"a:{x}")
    assert reg.create("a", 1) == "a:1"
    assert reg.names() == ["a"]


def test_unknown_provider_lists_available() -> None:
    reg: Registry[str] = Registry("demo")
    reg.register("a", lambda: "a")
    with pytest.raises(ConfigurationError, match="Unknown demo provider 'zzz'"):
        reg.create("zzz")


def test_missing_extra_gives_actionable_error() -> None:
    reg: Registry[object] = Registry("demo")
    reg.register("x", "nonexistent_pkg_xyz:build", extra="langgraph")
    with pytest.raises(ConfigurationError, match="uv sync --extra langgraph"):
        reg.create("x")


def test_decorator_registration() -> None:
    reg: Registry[int] = Registry("demo")

    @reg.provider("one")
    def _one() -> int:
        return 1

    assert reg.create("one") == 1


def test_invalid_provider_literal_rejected_by_settings() -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, orchestrator={"provider": "nope"})  # type: ignore[call-arg,arg-type]


def test_nested_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_LLM__PRIMARY_MODEL", "x/y")
    monkeypatch.setenv("AGENT_AUTH__ENABLED", "true")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.llm.primary_model == "x/y" and s.auth.enabled is True
