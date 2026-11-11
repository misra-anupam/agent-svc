from __future__ import annotations

from pathlib import Path

from agent_svc.adapters.llm.fake import FakeLLM
from agent_svc.adapters.prompts.file import FilePromptStore


async def test_system_prompt_comes_from_the_store_and_agent_info_reports_it(  # type: ignore[no-untyped-def]
    settings, make_client, tmp_path: Path
) -> None:
    store = FilePromptStore(tmp_path)
    await store.register("system", "You are a pirate.")
    await store.register("system", "You are a very formal butler.")
    await store.set_alias("system", "prod", "2")
    settings.prompts.path, settings.prompts.ref = str(tmp_path), "system@prod"
    primary = FakeLLM("fake/primary", reply="ok")
    async with make_client({"fake/primary": primary, "fake/secondary": FakeLLM()}) as c:
        info = (await c.get("/v1/agent")).json()
        await c.post("/v1/threads/t/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert info["prompt"] == "system:2"
    system_msgs = [m.content for call in primary.seen for m in call if m.role == "system"]
    assert system_msgs and all("butler" in m for m in system_msgs)
    assert settings.orchestrator.system_prompt != "You are a very formal butler."  # not mutated


async def test_missing_prompt_fails_startup(settings, make_client, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import pytest

    from agent_svc.domain.exceptions import NotFoundError

    settings.prompts.path, settings.prompts.ref = str(tmp_path), "nope@prod"
    with pytest.raises(NotFoundError):
        async with make_client():
            pass
