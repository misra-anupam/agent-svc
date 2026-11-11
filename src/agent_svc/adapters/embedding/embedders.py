from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from agent_svc.config.settings import Settings
from agent_svc.domain.exceptions import UpstreamError


class FakeEmbedder:
    """Deterministic hashed bag-of-words embedding: similar texts are close. For tests/local."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in text.lower().split():
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]


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


def build_fake(settings: Settings) -> FakeEmbedder:
    return FakeEmbedder(settings.memory.embedding_dim)


def build_litellm(settings: Settings) -> LiteLLMEmbedder:
    return LiteLLMEmbedder(settings.memory.embedding_model, settings.memory.embedding_dim)
