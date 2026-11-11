"""Guardrails served by sidecars over HTTP: Llama Guard, Prompt Guard, NeMo Guardrails.

No heavy SDKs in the API image: the models run in their own containers (compose profiles
``guardrails`` / ``guardrails-gpu``). Failures raise; ``GuardrailChain`` applies ``fail_mode``.
"""

from __future__ import annotations

from typing import Any, ClassVar

import httpx

from agent_svc.config.settings import HttpRailSettings, Settings
from agent_svc.domain.models import Principal
from agent_svc.domain.ports import GuardrailResult, GuardrailStage


class _HttpRail:
    name = ""

    def __init__(self, cfg: HttpRailSettings, client: httpx.AsyncClient | None = None) -> None:
        self._cfg = cfg
        headers = (
            {"Authorization": f"Bearer {cfg.api_key.get_secret_value()}"} if cfg.api_key else {}
        )
        self._client = client or httpx.AsyncClient(
            timeout=cfg.timeout_seconds, headers=headers, follow_redirects=False
        )

    async def _post(self, path: str, body: dict[str, Any]) -> Any:
        resp = await self._client.post(self._cfg.url.rstrip("/") + path, json=body)
        resp.raise_for_status()
        return resp.json()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def check(
        self, text: str, *, stage: GuardrailStage, principal: Principal, context: str = ""
    ) -> GuardrailResult:
        raise NotImplementedError


class LlamaGuardRail(_HttpRail):
    """Llama Guard behind any OpenAI-compatible ``/chat/completions`` (vLLM, Ollama, ...).

    The model answers ``safe`` or ``unsafe\\nS1,S9`` (hazard categories). Input and output stages.
    """

    name = "llama_guard"

    async def check(
        self, text: str, *, stage: GuardrailStage, principal: Principal, context: str = ""
    ) -> GuardrailResult:
        messages = [{"role": "user", "content": text}]
        if stage is GuardrailStage.OUTPUT:
            messages = [
                {"role": "user", "content": context or "(prompt unavailable)"},
                {"role": "assistant", "content": text},
            ]
        data = await self._post(
            "/chat/completions",
            {"model": self._cfg.model, "messages": messages, "temperature": 0, "max_tokens": 20},
        )
        verdict = str(data["choices"][0]["message"]["content"]).strip().lower()
        if verdict.startswith("unsafe"):
            cats = verdict.partition("\n")[2].strip().upper() or "UNSPECIFIED"
            return GuardrailResult("block", rail=self.name, reason=f"unsafe:{cats}")
        return GuardrailResult(text=text, rail=self.name)


class PromptGuardRail(_HttpRail):
    """Prompt Guard / any text-classification endpoint: ``POST {"inputs": text}``.

    Accepts ``[{"label","score"}]`` or ``[[{"label","score"}]]``. Blocks when an
    ``INJECTION`` / ``JAILBREAK`` label scores at or above ``threshold``. Input stage only.
    """

    name = "prompt_guard"
    _BAD: ClassVar[frozenset[str]] = frozenset({"injection", "jailbreak", "label_1", "label_2"})

    async def check(
        self, text: str, *, stage: GuardrailStage, principal: Principal, context: str = ""
    ) -> GuardrailResult:
        if stage is not GuardrailStage.INPUT:
            return GuardrailResult(text=text, rail=self.name)
        data = await self._post("", {"inputs": text})
        scores = data[0] if data and isinstance(data[0], list) else data
        worst = max(
            (s for s in scores if str(s["label"]).lower() in self._BAD),
            key=lambda s: s["score"],
            default=None,
        )
        if worst and worst["score"] >= self._cfg.threshold:
            return GuardrailResult(
                "block",
                rail=self.name,
                reason=f"{str(worst['label']).lower()}:{worst['score']:.2f}",
            )
        return GuardrailResult(text=text, rail=self.name)


class NemoRail(_HttpRail):
    """NeMo Guardrails server (``/v1/chat/completions`` with a rails config).

    The server runs its configured input/output rails; a rail that fires replaces the bot reply
    with its refusal text, which we recognise via ``refusal_markers``. Verify the marker against
    your Colang config: it is not a stable API contract.
    """

    name = "nemo"

    async def check(
        self, text: str, *, stage: GuardrailStage, principal: Principal, context: str = ""
    ) -> GuardrailResult:
        messages = [{"role": "user", "content": text}]
        if stage is GuardrailStage.OUTPUT:
            messages = [
                {"role": "user", "content": context or "(prompt unavailable)"},
                {"role": "assistant", "content": text},
                {"role": "user", "content": "(output check)"},
            ]
        data = await self._post(
            "/v1/chat/completions",
            {
                "model": self._cfg.model or "default",
                "config_id": self._cfg.config_id,
                "guardrails": {"config_id": self._cfg.config_id},
                "messages": messages,
            },
        )
        reply = str(data["choices"][0]["message"]["content"])
        if any(m.lower() in reply.lower() for m in self._cfg.refusal_markers):
            return GuardrailResult("block", rail=self.name, reason="rail_triggered")
        return GuardrailResult(text=text, rail=self.name)


def build_llama_guard(settings: Settings) -> LlamaGuardRail:
    return LlamaGuardRail(settings.guardrails.llama_guard)


def build_prompt_guard(settings: Settings) -> PromptGuardRail:
    return PromptGuardRail(settings.guardrails.prompt_guard)


def build_nemo(settings: Settings) -> NemoRail:
    return NemoRail(settings.guardrails.nemo)
