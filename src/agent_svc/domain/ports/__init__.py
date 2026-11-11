"""Ports: the interfaces every adapter implements (Dependency Inversion)."""

from agent_svc.domain.ports.agent import AgentRunner
from agent_svc.domain.ports.auth import Authenticator
from agent_svc.domain.ports.cache import Cache
from agent_svc.domain.ports.evaluation import EvalReport, Evaluator, GoldenCase
from agent_svc.domain.ports.guardrails import Guardrail, GuardrailResult, GuardrailStage
from agent_svc.domain.ports.lifecycle import (
    HealthCheck,
    ReadinessProvider,
    ReadinessReport,
    Resource,
)
from agent_svc.domain.ports.llm import LLMClient, LLMResponse
from agent_svc.domain.ports.memory import LongTermMemory, ThreadMemory, UserMemory
from agent_svc.domain.ports.prompts import PromptStore, PromptVersion
from agent_svc.domain.ports.ratelimit import RateLimiter, TokenBudget
from agent_svc.domain.ports.tools import ToolProvider
from agent_svc.domain.ports.tracing import Span, Tracer

__all__ = [
    "AgentRunner",
    "Authenticator",
    "Cache",
    "EvalReport",
    "Evaluator",
    "GoldenCase",
    "Guardrail",
    "GuardrailResult",
    "GuardrailStage",
    "HealthCheck",
    "LLMClient",
    "LLMResponse",
    "LongTermMemory",
    "PromptStore",
    "PromptVersion",
    "RateLimiter",
    "ReadinessProvider",
    "ReadinessReport",
    "Resource",
    "Span",
    "ThreadMemory",
    "TokenBudget",
    "ToolProvider",
    "Tracer",
    "UserMemory",
]
