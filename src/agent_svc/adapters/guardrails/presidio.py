"""Microsoft Presidio PII rail (extra: ``presidio`` + a spaCy model, e.g. en_core_web_sm).

Analysis is CPU-bound, so it runs in a worker thread. Anonymization replaces each entity with
``<ENTITY_TYPE>``. ``presidio_action = "block"`` rejects instead of redacting.
"""

from __future__ import annotations

import anyio

from agent_svc.config.settings import GuardrailSettings, Settings
from agent_svc.domain.exceptions import ConfigurationError
from agent_svc.domain.models import Principal
from agent_svc.domain.ports import GuardrailResult, GuardrailStage


class PresidioRail:
    name = "presidio"

    def __init__(self, cfg: GuardrailSettings) -> None:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine

        try:
            nlp = NlpEngineProvider(
                nlp_configuration={
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "en", "model_name": cfg.presidio_model}],
                }
            ).create_engine()
        except OSError as exc:  # spaCy model not installed
            raise ConfigurationError(
                f"spaCy model {cfg.presidio_model!r} is missing; run: "
                f"uv run python -m spacy download {cfg.presidio_model}"
            ) from exc
        self._analyzer = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])
        self._anonymizer = AnonymizerEngine()  # type: ignore[no-untyped-call]
        self._entities = cfg.presidio_entities
        self._threshold = cfg.presidio_threshold
        self._action = cfg.presidio_action

    def _scan(self, text: str) -> GuardrailResult:
        found = self._analyzer.analyze(
            text=text,
            language="en",
            entities=self._entities,
            score_threshold=self._threshold,
        )
        if not found:
            return GuardrailResult(text=text, rail=self.name)
        kinds = ",".join(sorted({r.entity_type for r in found}))
        if self._action == "block":
            return GuardrailResult("block", rail=self.name, reason=f"pii:{kinds}")
        result = self._anonymizer.anonymize(text=text, analyzer_results=found)  # type: ignore[arg-type]
        redacted: str = result.text
        return GuardrailResult("redact", redacted, self.name, f"pii:{kinds}")

    async def check(
        self, text: str, *, stage: GuardrailStage, principal: Principal, context: str = ""
    ) -> GuardrailResult:
        return await anyio.to_thread.run_sync(self._scan, text)


def build(settings: Settings) -> PresidioRail:
    return PresidioRail(settings.guardrails)
