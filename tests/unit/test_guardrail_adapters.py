"""Adapter tests: HTTP sidecars (mock transport), Presidio, Guardrails AI, LLM Guard (stubbed)."""

from __future__ import annotations

import sys
import types

import httpx
import pytest

from agent_svc.adapters.guardrails.http_rails import LlamaGuardRail, NemoRail, PromptGuardRail
from agent_svc.config.settings import GuardrailSettings, HttpRailSettings
from agent_svc.domain.models import Principal
from agent_svc.domain.ports import GuardrailStage

P = Principal("u")
IN, OUT = GuardrailStage.INPUT, GuardrailStage.OUTPUT


def client(handler) -> httpx.AsyncClient:  # type: ignore[no-untyped-def]
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ---- Llama Guard ---------------------------------------------------------------------------
def llama(reply: str, seen: list[dict]) -> LlamaGuardRail:  # type: ignore[type-arg]
    def handler(req: httpx.Request) -> httpx.Response:
        import json

        seen.append(json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})

    return LlamaGuardRail(
        HttpRailSettings(url="http://lg/v1", model="llama-guard"), client(handler)
    )


async def test_llama_guard_unsafe_blocks_with_categories() -> None:
    res = await llama("unsafe\nS1,S9", []).check("bad", stage=IN, principal=P)
    assert res.action == "block" and res.reason == "unsafe:S1,S9"


async def test_llama_guard_safe_allows_and_output_stage_sends_the_conversation() -> None:
    seen: list[dict] = []  # type: ignore[type-arg]
    res = await llama("safe", seen).check("answer", stage=OUT, principal=P, context="question")
    assert res.action == "allow"
    assert [m["role"] for m in seen[0]["messages"]] == ["user", "assistant"]
    assert seen[0]["messages"][0]["content"] == "question"


# ---- Prompt Guard --------------------------------------------------------------------------
def pg(scores: object, threshold: float = 0.5) -> PromptGuardRail:
    return PromptGuardRail(
        HttpRailSettings(url="http://pg/classify", threshold=threshold),
        client(lambda r: httpx.Response(200, json=scores)),
    )


@pytest.mark.parametrize(
    "payload",
    [
        [[{"label": "INJECTION", "score": 0.97}, {"label": "BENIGN", "score": 0.02}]],
        [{"label": "JAILBREAK", "score": 0.9}, {"label": "BENIGN", "score": 0.1}],
    ],
)
async def test_prompt_guard_blocks_above_threshold(payload: object) -> None:
    assert (await pg(payload).check("x", stage=IN, principal=P)).action == "block"


async def test_prompt_guard_allows_below_threshold_and_ignores_output_stage() -> None:
    low = [[{"label": "INJECTION", "score": 0.2}, {"label": "BENIGN", "score": 0.8}]]
    assert (await pg(low).check("x", stage=IN, principal=P)).action == "allow"
    high = [[{"label": "INJECTION", "score": 0.99}]]
    assert (await pg(high).check("x", stage=OUT, principal=P)).action == "allow"


# ---- NeMo Guardrails -----------------------------------------------------------------------
def nemo(reply: str) -> NemoRail:
    return NemoRail(
        HttpRailSettings(url="http://nemo:8000", config_id="cfg"),
        client(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})),
    )


async def test_nemo_refusal_marker_means_blocked() -> None:
    res = await nemo("I'm sorry, I can't respond to that.").check("x", stage=IN, principal=P)
    assert res.action == "block" and res.reason == "rail_triggered"
    assert (await nemo("Sure, here you go").check("x", stage=IN, principal=P)).action == "allow"


async def test_http_error_propagates_so_the_chain_can_apply_fail_mode() -> None:
    rail = PromptGuardRail(
        HttpRailSettings(url="http://pg"), client(lambda r: httpx.Response(503, text="down"))
    )
    with pytest.raises(httpx.HTTPStatusError):
        await rail.check("x", stage=IN, principal=P)


# ---- Presidio ------------------------------------------------------------------------------
def _presidio(action: str = "redact"):  # type: ignore[no-untyped-def]
    pytest.importorskip("presidio_analyzer")
    from agent_svc.adapters.guardrails.presidio import PresidioRail
    from agent_svc.domain.exceptions import ConfigurationError

    try:
        return PresidioRail(GuardrailSettings(presidio_action=action))  # type: ignore[arg-type]
    except ConfigurationError:
        pytest.skip("spaCy model en_core_web_sm not installed")


@pytest.mark.parametrize(
    ("text", "gone"),
    [
        ("mail me at jo.smith@example.com", "jo.smith@example.com"),
        ("card 4111 1111 1111 1111 thanks", "4111 1111 1111 1111"),
        ("my SSN is 536-90-4399", "536-90-4399"),
    ],
)
async def test_presidio_redacts_pii(text: str, gone: str) -> None:
    res = await _presidio().check(text, stage=OUT, principal=P)
    assert res.action == "redact" and gone not in res.text and "<" in res.text


async def test_presidio_block_mode_and_clean_text() -> None:
    assert (
        await _presidio("block").check("jo@example.com", stage=IN, principal=P)
    ).action == "block"
    assert (
        await _presidio().check("What is the capital of France?", stage=IN, principal=P)
    ).action == "allow"


# ---- Guardrails AI -------------------------------------------------------------------------
def _gai():  # type: ignore[no-untyped-def]
    pytest.importorskip("guardrails")
    from agent_svc.adapters.guardrails.guardrails_ai import GuardrailsAiRail

    return GuardrailsAiRail(
        GuardrailSettings(
            guardrails_ai_validators=["tests.fixtures.guard_validators:NoForbiddenWord"]
        )
    )


async def test_guardrails_ai_validator_blocks_and_telemetry_is_off() -> None:
    from guardrails.settings import settings as gr

    rail = _gai()
    assert gr.rc.enable_metrics is False  # the SDK phones home by default
    assert rail._guard._hub_telemetry._enabled is False
    assert (await rail.check("a FORBIDDEN thing", stage=OUT, principal=P)).action == "block"
    assert (await rail.check("a fine thing", stage=OUT, principal=P)).action == "allow"


async def test_guardrails_ai_requires_validators() -> None:
    pytest.importorskip("guardrails")
    from agent_svc.adapters.guardrails.guardrails_ai import GuardrailsAiRail
    from agent_svc.domain.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match="validators"):
        GuardrailsAiRail(GuardrailSettings())


# ---- LLM Guard (stubbed: the real package is torch-heavy and archived upstream) ---------------
@pytest.fixture
def fake_llm_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    lib = types.ModuleType("llm_guard")
    inp = types.ModuleType("llm_guard.input_scanners")
    out = types.ModuleType("llm_guard.output_scanners")
    inp.PromptInjection = type("PromptInjection", (), {})  # type: ignore[attr-defined]
    out.Sensitive = type("Sensitive", (), {})  # type: ignore[attr-defined]

    def scan_prompt(scanners, text):  # type: ignore[no-untyped-def]
        bad = "attack" in text
        return text.replace("secret", "***"), {"PromptInjection": not bad}, {}

    def scan_output(scanners, prompt, text):  # type: ignore[no-untyped-def]
        return text, {"Sensitive": True}, {}

    lib.scan_prompt, lib.scan_output = scan_prompt, scan_output  # type: ignore[attr-defined]
    for name, mod in {
        "llm_guard": lib,
        "llm_guard.input_scanners": inp,
        "llm_guard.output_scanners": out,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)


async def test_llm_guard_adapter_maps_scanner_results(fake_llm_guard: None) -> None:
    from agent_svc.adapters.guardrails.llm_guard import LlmGuardRail

    rail = LlmGuardRail(GuardrailSettings())
    blocked = await rail.check("an attack", stage=IN, principal=P)
    assert blocked.action == "block" and blocked.reason == "scanner:PromptInjection"
    sanitized = await rail.check("my secret", stage=IN, principal=P)
    assert sanitized.action == "redact" and sanitized.text == "my ***"
    assert (await rail.check("fine", stage=OUT, principal=P, context="q")).action == "allow"


def test_missing_extra_is_reported_with_the_uv_command(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_svc.adapters import registries as reg
    from agent_svc.config.settings import Settings
    from agent_svc.domain.exceptions import ConfigurationError

    monkeypatch.setitem(sys.modules, "llm_guard", None)  # force ImportError
    with pytest.raises(ConfigurationError, match="uv pip install llm-guard"):
        reg.guardrails.create("llm_guard", Settings(_env_file=None))  # type: ignore[call-arg]
