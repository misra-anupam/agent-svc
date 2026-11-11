"""Ports: the interfaces every adapter implements (Dependency Inversion)."""

from agent_svc.domain.ports.agent import AgentRunner
from agent_svc.domain.ports.auth import Authenticator
from agent_svc.domain.ports.cache import Cache
from agent_svc.domain.ports.embedding import Embedder
from agent_svc.domain.ports.evaluation import (
    EvalRecord,
    EvalReport,
    Evaluator,
    GoldenCase,
    Threshold,
)
from agent_svc.domain.ports.guardrails import Guardrail, GuardrailResult, GuardrailStage
from agent_svc.domain.ports.lifecycle import (
    HealthCheck,
    ReadinessProvider,
    ReadinessReport,
    Resource,
)
from agent_svc.domain.ports.llm import LLMChunk, LLMClient, LLMResponse
from agent_svc.domain.ports.memory import LongTermMemory, ThreadMemory, UserMemory
from agent_svc.domain.ports.prompts import PromptStore, PromptVersion
from agent_svc.domain.ports.ratelimit import (
    ConcurrencyGate,
    LimitDecision,
    RateLimiter,
    TokenBudget,
)
from agent_svc.domain.ports.runs import RunBuffer
from agent_svc.domain.ports.tools import ToolProvider, step_kind_for
from agent_svc.domain.ports.tracing import Span, Tracer

__all__ = [
    "AgentRunner",
    "Authenticator",
    "Cache",
    "ConcurrencyGate",
    "Embedder",
    "EvalRecord",
    "EvalReport",
    "Evaluator",
    "GoldenCase",
    "Guardrail",
    "GuardrailResult",
    "GuardrailStage",
    "HealthCheck",
    "LLMChunk",
    "LLMClient",
    "LLMResponse",
    "LimitDecision",
    "LongTermMemory",
    "PromptStore",
    "PromptVersion",
    "RateLimiter",
    "ReadinessProvider",
    "ReadinessReport",
    "Resource",
    "RunBuffer",
    "Span",
    "ThreadMemory",
    "Threshold",
    "TokenBudget",
    "ToolProvider",
    "Tracer",
    "UserMemory",
    "step_kind_for",
]
