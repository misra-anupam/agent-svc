from __future__ import annotations

from pathlib import Path

from tests.scripted import ScriptedModel

from agent_svc.adapters.prompts.mlflow import MlflowPromptStore


async def test_system_prompt_comes_from_the_store_and_agent_info_reports_it(  # type: ignore[no-untyped-def]
    settings, make_client, tmp_path: Path
) -> None:
    uri = f"sqlite:///{tmp_path}/m.db"
    store = MlflowPromptStore(uri)
    await store.register("system", "You are a pirate.")
    await store.register("system", "You are a very formal butler.")
    await store.set_alias("system", "prod", "2")
    settings.prompts.mlflow_tracking_uri, settings.prompts.ref = uri, "system@prod"
    primary = ScriptedModel("test/primary", reply="ok")
    async with make_client({"test/primary": primary, "test/secondary": ScriptedModel()}) as c:
        info = (await c.get("/v1/agent")).json()
        await c.post("/v1/threads/t/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert info["prompt"] == "system:2"
    system_msgs = [m.content for call in primary.seen for m in call if m.role == "system"]
    assert system_msgs and all("butler" in m for m in system_msgs)
    assert settings.orchestrator.system_prompt != "You are a very formal butler."  # not mutated


async def test_missing_prompt_fails_startup(settings, make_client, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import pytest

    from agent_svc.domain.exceptions import NotFoundError

    settings.prompts.mlflow_tracking_uri = f"sqlite:///{tmp_path}/m.db"
    settings.prompts.ref = "nope@prod"
    with pytest.raises(NotFoundError):
        async with make_client():
            pass
