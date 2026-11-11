from __future__ import annotations

from collections.abc import Sequence

from agent_svc.config.settings import Settings
from agent_svc.domain.exceptions import UpstreamError


class LiteLLMEmbedder:
    def __init__(self, model: str, dim: int) -> None:
        self._model, self.dim = model, dim

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        import litellm

        try:
            resp = await litellm.aembedding(model=self._model, input=list(texts))
        except Exception as exc:
            raise UpstreamError(f"embedding failed: {exc}") from exc
        return [list(d["embedding"]) for d in resp.data]


def build_litellm(settings: Settings) -> LiteLLMEmbedder:
    return LiteLLMEmbedder(settings.memory.embedding_model, settings.memory.embedding_dim)
