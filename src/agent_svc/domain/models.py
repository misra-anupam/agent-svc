"""Internal contracts as frozen, slotted dataclasses (pydantic lives at the API edge)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str
    name: str | None = None


@dataclass(frozen=True, slots=True)
class Principal:
    subject: str
    tier: str = "free"
    tenant: str = "default"
    scopes: frozenset[str] = frozenset()
    ip: str | None = None
    anonymous: bool = False

    @property
    def rate_key(self) -> str:
        return f"user:{self.subject}" if not self.anonymous else f"ip:{self.ip or 'unknown'}"


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cost_usd + other.cost_usd,
        )


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    source: str = "local"  # "local" | "mcp:<server>"


@dataclass(frozen=True, slots=True)
class AgentRequest:
    run_id: str
    thread_id: str
    principal: Principal
    messages: tuple[Message, ...]
    agent_name: str
    agent_version: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def last_user_text(self) -> str:
        for m in reversed(self.messages):
            if m.role == "user":
                return m.content
        return ""


class StepKind(StrEnum):
    TOOL = "tool"
    RAG = "rag"
    AGENT = "agent"
    MCP = "mcp"
    GUARDRAIL = "guardrail"
    LLM = "llm"


# --- streamed events (framework-neutral) -----------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentEvent:
    type: ClassVar[str] = "event"
    run_id: str


@dataclass(frozen=True, slots=True, kw_only=True)
class RunStarted(AgentEvent):
    type: ClassVar[str] = "run.started"
    thread_id: str
    agent_name: str
    agent_version: str


@dataclass(frozen=True, slots=True, kw_only=True)
class MessageDelta(AgentEvent):
    type: ClassVar[str] = "message.delta"
    text: str


@dataclass(frozen=True, slots=True, kw_only=True)
class StepStarted(AgentEvent):
    type: ClassVar[str] = "step.started"
    step_id: str
    kind: StepKind
    name: str
    parent_id: str | None = None
    input: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class StepFinished(AgentEvent):
    type: ClassVar[str] = "step.finished"
    step_id: str
    ok: bool = True
    output_summary: str = ""
    duration_ms: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class UsageReported(AgentEvent):
    type: ClassVar[str] = "usage"
    usage: Usage
    model: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class MessageCompleted(AgentEvent):
    type: ClassVar[str] = "message.completed"
    text: str


@dataclass(frozen=True, slots=True, kw_only=True)
class RunError(AgentEvent):
    type: ClassVar[str] = "run.error"
    code: str
    message: str
    retryable: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class RunCompleted(AgentEvent):
    type: ClassVar[str] = "run.completed"
    usage: Usage = Usage()
    cached: bool = False
