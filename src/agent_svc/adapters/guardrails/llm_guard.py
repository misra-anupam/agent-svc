"""LLM Guard rail. DEPRECATED.

Upstream (protectai/llm-guard) is reported archived, and pulls in torch/transformers. Kept for
existing users. It has no uv extra because its pins conflict with the shared lock: install it
with ``uv pip install llm-guard``. Prefer the ``prompt_guard`` / ``llama_guard`` / ``nemo``
sidecars. Scanners are chosen by class name from ``llm_guard.input_scanners`` /
``llm_guard.output_scanners``.
"""

from __future__ import annotations

import importlib
from typing import Any

import anyio
import structlog

from agent_svc.config.settings import GuardrailSettings, Settings
from agent_svc.domain.exceptions import ConfigurationError
from agent_svc.domain.models import Principal
from agent_svc.domain.ports import GuardrailResult, GuardrailStage

log = structlog.get_logger()


def _scanners(module: str, names: list[str]) -> list[Any]:
    mod = importlib.import_module(module)
    try:
        return [getattr(mod, n)() for n in names]
    except AttributeError as exc:
        raise ConfigurationError(f"unknown llm_guard scanner in {names}: {exc}") from exc


class LlmGuardRail:
    name = "llm_guard"

    def __init__(self, cfg: GuardrailSettings) -> None:
        try:
            import llm_guard
        except ImportError as exc:
            raise ImportError(
                "llm-guard is not installed. Its pins conflict with this project's lock, so it has "
                "no uv extra: install it separately with `uv pip install llm-guard`"
            ) from exc

        log.warning("llm_guard_deprecated", detail="upstream is archived; prefer a sidecar rail")
        self._lib = llm_guard
        self._input = _scanners("llm_guard.input_scanners", cfg.llm_guard_input_scanners)
        self._output = _scanners("llm_guard.output_scanners", cfg.llm_guard_output_scanners)

    def _scan(self, text: str, stage: GuardrailStage, context: str) -> GuardrailResult:
        if stage is GuardrailStage.INPUT:
            clean, valid, _ = self._lib.scan_prompt(self._input, text)
        else:
            clean, valid, _ = self._lib.scan_output(self._output, context, text)
        failed = sorted(k for k, ok in valid.items() if not ok)
        if failed:
            return GuardrailResult("block", rail=self.name, reason="scanner:" + ",".join(failed))
        if clean != text:
            return GuardrailResult("redact", clean, self.name, "sanitized")
        return GuardrailResult(text=text, rail=self.name)

    async def check(
        self, text: str, *, stage: GuardrailStage, principal: Principal, context: str = ""
    ) -> GuardrailResult:
        return await anyio.to_thread.run_sync(self._scan, text, stage, context)


def build(settings: Settings) -> LlmGuardRail:
    return LlmGuardRail(settings.guardrails)
