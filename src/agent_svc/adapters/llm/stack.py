"""Builds the resilient LLM stack from settings:

Metered( Fallback( Retrying(Breaker(primary)), Retrying(Breaker(secondary)) ) )
"""

from __future__ import annotations

from collections.abc import Callable

from agent_svc.adapters.llm.decorators import BreakerLLM, FallbackLLM, MeteredLLM, RetryingLLM
from agent_svc.adapters.resilience.circuit_breaker import CircuitBreaker
from agent_svc.adapters.resilience.retry import make_retrying
from agent_svc.config.settings import LLMSettings
from agent_svc.domain.ports import LLMClient


def _harden(base: LLMClient, name: str, s: LLMSettings) -> LLMClient:
    breaker = CircuitBreaker(
        name, failure_threshold=s.breaker_failure_threshold, reset_seconds=s.breaker_reset_seconds
    )
    return RetryingLLM(
        BreakerLLM(base, breaker), lambda: make_retrying(name, max_attempts=s.max_retries)
    )


def build_llm_stack(s: LLMSettings, factory: Callable[[str], LLMClient]) -> LLMClient:
    primary = _harden(factory(s.primary_model), s.primary_model, s)
    if not s.secondary_model:
        return MeteredLLM(primary, s.primary_model)
    secondary = _harden(factory(s.secondary_model), s.secondary_model, s)
    return MeteredLLM(
        FallbackLLM(primary, secondary, primary_name=s.primary_model), s.primary_model
    )
