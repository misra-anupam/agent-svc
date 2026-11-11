from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class MessageIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant", "system"] = "user"
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: list[MessageIn] = Field(min_length=1)


class UsageOut(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class _Base(BaseModel):
    run_id: str


class RunStartedEvent(_Base):
    type: Literal["run.started"] = "run.started"
    thread_id: str
    agent_name: str
    agent_version: str


class MessageDeltaEvent(_Base):
    type: Literal["message.delta"] = "message.delta"
    text: str


class StepStartedEvent(_Base):
    type: Literal["step.started"] = "step.started"
    step_id: str
    kind: Literal["tool", "rag", "agent", "mcp", "guardrail", "llm"]
    name: str
    parent_id: str | None = None
    input: dict[str, object] = Field(default_factory=dict)


class StepFinishedEvent(_Base):
    type: Literal["step.finished"] = "step.finished"
    step_id: str
    ok: bool = True
    output_summary: str = ""
    duration_ms: int = 0


class UsageEvent(_Base):
    type: Literal["usage"] = "usage"
    usage: UsageOut
    model: str = ""


class MessageCompletedEvent(_Base):
    type: Literal["message.completed"] = "message.completed"
    text: str


class RunErrorEvent(_Base):
    type: Literal["run.error"] = "run.error"
    code: str
    message: str
    retryable: bool = False


class RunCompletedEvent(_Base):
    type: Literal["run.completed"] = "run.completed"
    usage: UsageOut = Field(default_factory=UsageOut)
    cached: bool = False


SseEvent = Annotated[
    RunStartedEvent
    | MessageDeltaEvent
    | StepStartedEvent
    | StepFinishedEvent
    | UsageEvent
    | MessageCompletedEvent
    | RunErrorEvent
    | RunCompletedEvent,
    Field(discriminator="type"),
]

EVENT_MODELS: dict[str, type[BaseModel]] = {
    "run.started": RunStartedEvent,
    "message.delta": MessageDeltaEvent,
    "step.started": StepStartedEvent,
    "step.finished": StepFinishedEvent,
    "usage": UsageEvent,
    "message.completed": MessageCompletedEvent,
    "run.error": RunErrorEvent,
    "run.completed": RunCompletedEvent,
}
