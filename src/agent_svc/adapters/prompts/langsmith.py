"""LangSmith prompt hub store (extra: ``langsmith`` + ``langchain-core``).

Versions are LangSmith commit hashes; aliases are commit tags (``name:tag``). LangSmith has no
public "move tag to commit" call, so ``set_alias`` is unsupported: manage tags in LangSmith, or
pass ``alias=`` to ``get``. NOT verified against the live service (needs an account).
"""

from __future__ import annotations

from typing import Any

import anyio

from agent_svc.config.settings import Settings
from agent_svc.domain.exceptions import ConfigurationError, NotFoundError
from agent_svc.domain.ports import PromptVersion


class LangSmithPromptStore:
    name = "langsmith"

    def __init__(self, client: Any) -> None:
        self._client = client

    def _get(self, name: str, version: str | None, alias: str | None) -> PromptVersion:
        ident = f"{name}:{alias or version or 'latest'}"
        try:
            commit = self._client.pull_prompt_commit(ident)
            prompt = self._client.pull_prompt(ident)
        except Exception as exc:
            raise NotFoundError(f"prompt {ident!r} not found: {exc}") from exc
        template = getattr(prompt, "template", None)
        if not isinstance(template, str):
            raise ConfigurationError(
                f"LangSmith prompt {ident!r} is not a plain string template "
                f"({type(prompt).__name__}); only string prompts are supported"
            )
        return PromptVersion(name, str(commit.commit_hash), template)

    def _register(self, name: str, template: str) -> PromptVersion:
        from langchain_core.prompts import PromptTemplate

        self._client.push_prompt(name, object=PromptTemplate.from_template(template))
        return self._get(name, None, None)

    async def get(
        self, name: str, *, version: str | None = None, alias: str | None = None
    ) -> PromptVersion:
        return await anyio.to_thread.run_sync(self._get, name, version, alias)

    async def register(self, name: str, template: str) -> PromptVersion:
        return await anyio.to_thread.run_sync(self._register, name, template)

    async def set_alias(self, name: str, alias: str, version: str) -> None:
        raise ConfigurationError("LangSmith aliases are commit tags: manage them in LangSmith")


def build(settings: Settings) -> LangSmithPromptStore:
    from langsmith import Client

    key = settings.prompts.langsmith_api_key
    return LangSmithPromptStore(Client(api_key=key.get_secret_value() if key else None))
