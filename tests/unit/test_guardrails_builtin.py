from __future__ import annotations

import pytest
from tests.guardrails.dataset import Case, load

from agent_svc.adapters.guardrails.builtin import BuiltinRail
from agent_svc.config.settings import GuardrailSettings, Settings
from agent_svc.domain.models import Principal
from agent_svc.domain.ports import GuardrailStage

PROMPT = Settings(_env_file=None).orchestrator.system_prompt  # type: ignore[call-arg]
CASES = load()


@pytest.fixture
def rail() -> BuiltinRail:
    return BuiltinRail(GuardrailSettings(), system_prompt=PROMPT)


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
async def test_builtin_adversarial_suite(rail: BuiltinRail, case: Case) -> None:
    res = await rail.check(case.text, stage=GuardrailStage(case.stage), principal=Principal("u"))
    assert res.action == case.expect, (case.owasp, res)
    if case.expect == "redact":
        assert res.text != case.text and "[REDACTED:" in res.text
    if case.expect == "allow":
        assert res.text == case.text


async def test_redaction_keeps_surrounding_text(rail: BuiltinRail) -> None:
    res = await rail.check(
        "key AKIAIOSFODNN7EXAMPLE end", stage=GuardrailStage.INPUT, principal=Principal("u")
    )
    assert res.text == "key [REDACTED:aws_access_key] end"


async def test_short_prompt_is_never_flagged_as_leak() -> None:
    rail = BuiltinRail(GuardrailSettings(), system_prompt="Be nice.")
    res = await rail.check("Be nice.", stage=GuardrailStage.OUTPUT, principal=Principal("u"))
    assert res.action == "allow"
