"""MLflow Prompt Registry store (extra: ``mlflow_genai``; needs the full ``mlflow`` package).

Versions are MLflow's integer versions; aliases are MLflow aliases (``prompts:/name@alias``).
The registry client is process-global in MLflow, so the tracking URI is set once here.
"""

from __future__ import annotations

import anyio

from agent_svc.config.settings import Settings
from agent_svc.domain.exceptions import NotFoundError
from agent_svc.domain.ports import PromptVersion


class MlflowPromptStore:
    name = "mlflow"

    def __init__(self, tracking_uri: str) -> None:
        import mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self._genai = mlflow.genai

    def _get(self, name: str, version: str | None, alias: str | None) -> PromptVersion:
        uri = (
            f"prompts:/{name}@{alias}"
            if alias
            else f"prompts:/{name}/{version}"
            if version
            else f"prompts:/{name}@latest"
        )
        try:
            p = self._genai.load_prompt(uri)
        except Exception as exc:  # mlflow raises MlflowException subclasses without a stable base
            raise NotFoundError(f"prompt {uri} not found: {exc}") from exc
        return PromptVersion(name, str(p.version), str(p.template))

    def _register(self, name: str, template: str) -> PromptVersion:
        p = self._genai.register_prompt(name=name, template=template)
        return PromptVersion(name, str(p.version), str(p.template))

    async def get(
        self, name: str, *, version: str | None = None, alias: str | None = None
    ) -> PromptVersion:
        return await anyio.to_thread.run_sync(self._get, name, version, alias)

    async def register(self, name: str, template: str) -> PromptVersion:
        return await anyio.to_thread.run_sync(self._register, name, template)

    async def set_alias(self, name: str, alias: str, version: str) -> None:
        def run() -> None:
            try:
                self._genai.set_prompt_alias(name, alias=alias, version=int(version))
            except Exception as exc:  # unknown prompt or version
                raise NotFoundError(f"prompt {name} version {version} not found: {exc}") from exc

        await anyio.to_thread.run_sync(run)


def build(settings: Settings) -> MlflowPromptStore:
    return MlflowPromptStore(settings.prompts.mlflow_tracking_uri)
