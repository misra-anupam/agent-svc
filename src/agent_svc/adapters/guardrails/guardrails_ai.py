"""Guardrails AI rail (extra: ``guardrails_ai``).

Validators are configured as ``module:attr`` factories (a Validator class or zero-arg callable),
so Hub validators and your own both work. **Telemetry is switched off**: by default the SDK
exports traces to Guardrails' servers.
"""

from __future__ import annotations

import importlib
from typing import Any

import anyio

from agent_svc.config.settings import GuardrailSettings, Settings
from agent_svc.domain.exceptions import ConfigurationError
from agent_svc.domain.models import Principal
from agent_svc.domain.ports import GuardrailResult, GuardrailStage


def _load(ref: str) -> Any:
    module, _, attr = ref.partition(":")
    if not attr:
        raise ConfigurationError(f"guardrails_ai validator {ref!r} must look like 'module:attr'")
    return getattr(importlib.import_module(module), attr)


class GuardrailsAiRail:
    name = "guardrails_ai"

    def __init__(self, cfg: GuardrailSettings) -> None:
        from guardrails.settings import settings as gr_settings

        from guardrails import Guard, OnFailAction

        gr_settings.rc.enable_metrics = False  # no telemetry to a third party
        if not cfg.guardrails_ai_validators:
            raise ConfigurationError("guardrails_ai selected but guardrails_ai_validators is empty")
        validators = []
        for ref in cfg.guardrails_ai_validators:
            v = _load(ref)
            validators.append(v(on_fail=OnFailAction.EXCEPTION) if isinstance(v, type) else v())
        self._guard = Guard().use(*validators)
        # Guard() re-reads ~/.guardrailsrc (metrics default ON), so switch telemetry off again
        # after construction, on the guard, its hub tracer and every validator.
        gr_settings.rc.enable_metrics = False
        self._guard._configure_hub_telemtry(False)
        for v in validators:
            v._disable_telemetry = True

    def _validate(self, text: str) -> GuardrailResult:
        from guardrails.errors import ValidationError

        try:
            self._guard.validate(text)
        except ValidationError as exc:
            return GuardrailResult("block", rail=self.name, reason=str(exc)[:200])
        return GuardrailResult(text=text, rail=self.name)

    async def check(
        self, text: str, *, stage: GuardrailStage, principal: Principal, context: str = ""
    ) -> GuardrailResult:
        return await anyio.to_thread.run_sync(self._validate, text)


def build(settings: Settings) -> GuardrailsAiRail:
    return GuardrailsAiRail(settings.guardrails)
