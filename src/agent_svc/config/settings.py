"""Settings: one small sub-settings model per pluggable component.

Selectors are Literals, so an invalid provider fails at startup. Env vars use the
``AGENT_`` prefix and ``__`` as the nesting delimiter, e.g. ``AGENT_LLM__PRIMARY_MODEL``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "test", "staging", "production"]


class AppSettings(BaseModel):
    name: str = "agent-svc"
    environment: Environment = "local"
    log_level: str = "INFO"
    log_json: bool = True
    cors_origins: list[str] = Field(default_factory=list)
    agent_name: str = "default"
    agent_version: str = "1.0.0"
    shutdown_grace_seconds: float = 20.0
    max_messages_per_request: int = 50
    max_message_chars: int = 20_000
    sse_heartbeat_seconds: float = 15.0
    trusted_proxy_hops: int = 0  # >0: take client IP from X-Forwarded-For (behind N proxies)
    run_buffer: Literal["inmemory", "redis"] = "inmemory"
    run_buffer_ttl_seconds: int = 3_600
    detach_on_disconnect: bool = False  # True: runs finish after disconnect so clients can resume


class RedisSettings(BaseModel):
    enabled: bool = True
    url: str = "redis://localhost:6379/0"
    required: bool = False  # if True, startup/readiness fails when Redis is down


class PostgresSettings(BaseModel):
    enabled: bool = False
    dsn: SecretStr = SecretStr("postgresql://agent:agent@localhost:5432/agent")
    pool_min: int = 0
    pool_max: int = 10
    required: bool = False


class LLMSettings(BaseModel):
    provider: Literal["litellm", "fake"] = "litellm"
    primary_model: str = "anthropic/claude-sonnet-4-5"
    secondary_model: str | None = None
    timeout_seconds: float = 60.0
    max_retries: int = 3
    breaker_failure_threshold: int = 5
    breaker_reset_seconds: float = 30.0


class OrchestratorSettings(BaseModel):
    provider: Literal["langgraph", "adk", "deepagents", "maf", "crewai", "echo"] = "echo"
    system_prompt: str = "You are a helpful assistant. Use tools when they help."
    max_iterations: int = 8  # cap on model/tool loops per run (OWASP LLM10)


class ToolsSettings(BaseModel):
    local: list[Literal["calculator", "current_time"]] = []
    allowlist: list[str] | None = None  # glob patterns; None = every discovered tool
    timeout_seconds: float = 30.0
    max_result_chars: int = 8_000


class TracingSettings(BaseModel):
    provider: Literal["none", "otel", "mlflow", "langfuse"] = "none"
    service_name: str = "agent-svc"
    # Prompts/answers are NOT recorded unless explicitly enabled (privacy / OWASP LLM02).
    capture_content: bool = False
    # otel
    otlp_endpoint: str | None = None  # e.g. http://otel-collector:4318
    otlp_headers: dict[str, str] = Field(default_factory=dict)
    sample_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    # mlflow
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "agent-svc"
    # langfuse
    langfuse_host: str | None = None
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None


class PromptStoreSettings(BaseModel):
    provider: Literal["file", "mlflow", "langsmith"] = "file"
    path: str = "prompts"


class EvalSettings(BaseModel):
    provider: Literal["none", "mlflow_genai", "langsmith", "adk_eval"] = "none"
    golden_dir: str = "evals/golden"


class HttpRailSettings(BaseModel):
    """A guardrail served by a sidecar over HTTP (Llama Guard, Prompt Guard, NeMo Guardrails)."""

    url: str = ""
    api_key: SecretStr | None = None
    model: str = ""
    timeout_seconds: float = 5.0
    threshold: float = 0.5  # classifier score at/above which the rail blocks
    config_id: str = ""  # NeMo Guardrails config
    refusal_markers: list[str] = ["I'm sorry, I can't respond to that."]  # NeMo: bot refusal text


class TierPolicy(BaseModel):
    """Per-tier override. ``None`` lists inherit the global ``enabled_*`` selection."""

    enabled: bool = True
    input: list[str] | None = None
    output: list[str] | None = None


class GuardrailSettings(BaseModel):
    enabled: bool = True
    enabled_input: list[
        Literal["builtin", "presidio", "prompt_guard", "guardrails_ai", "nemo", "llm_guard"]
    ] = ["builtin"]
    enabled_output: list[
        Literal["builtin", "presidio", "llama_guard", "guardrails_ai", "nemo", "llm_guard"]
    ] = ["builtin"]
    # A rail that errors or times out: "closed" blocks the request, "open" lets it through.
    fail_mode: Literal["closed", "open"] = "closed"
    max_input_chars: int = 8_000
    rail_timeout_seconds: float = 10.0
    # "buffer": hold the answer until the output rails pass (nothing unsafe ever leaves).
    # "stream": pass deltas through and re-check every ``stream_check_chars`` (lower latency,
    # but a prefix may already be sent when a rail blocks mid-answer).
    output_mode: Literal["buffer", "stream"] = "buffer"
    stream_check_chars: int = 400
    tier_policy: dict[str, TierPolicy] = Field(default_factory=dict)
    # presidio
    presidio_model: str = "en_core_web_sm"
    presidio_entities: list[str] = [
        "EMAIL_ADDRESS",
        "CREDIT_CARD",
        "PHONE_NUMBER",
        "US_SSN",
        "IBAN_CODE",
        "IP_ADDRESS",
    ]
    presidio_threshold: float = 0.5
    presidio_action: Literal["redact", "block"] = "redact"
    # guardrails_ai: "module:attr" factories returning a Validator (Hub validators work too)
    guardrails_ai_validators: list[str] = []
    # llm_guard (archived upstream; kept for existing users): scanner class names
    llm_guard_input_scanners: list[str] = ["PromptInjection"]
    llm_guard_output_scanners: list[str] = ["Sensitive"]
    llama_guard: HttpRailSettings = Field(default_factory=HttpRailSettings)
    prompt_guard: HttpRailSettings = Field(default_factory=HttpRailSettings)
    nemo: HttpRailSettings = Field(default_factory=HttpRailSettings)


class MemorySettings(BaseModel):
    """ "none" thread memory = stateless: clients send the full history on each request."""

    thread_provider: Literal["none", "inmemory", "redis"] = "none"
    user_provider: Literal["none", "inmemory", "postgres"] = "none"
    long_term_provider: Literal["none", "inmemory", "pgvector"] = "none"
    thread_ttl_seconds: int = 86_400
    max_thread_messages: int = 200
    max_context_tokens: int = 8_000
    long_term_top_k: int = 4
    long_term_write: bool = True
    embedding_provider: Literal["fake", "litellm"] = "fake"
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dim: int = 1536


class CacheSettings(BaseModel):
    provider: Literal["none", "inmemory", "redis"] = "none"
    ttl_seconds: int = 3_600
    scope: Literal["user", "tenant", "global"] = "user"  # who may share a cached answer
    max_entries: int = 1_000  # inmemory only


class TierLimit(BaseModel):
    requests_per_minute: int
    burst: int
    daily_token_budget: int
    max_concurrent: int = 2


class RateLimitSettings(BaseModel):
    provider: Literal["none", "redis", "inmemory"] = "none"
    mode: Literal["reject", "throttle"] = "reject"
    max_throttle_delay_ms: int = 2_000
    fail_open: bool = True
    tiers: dict[str, TierLimit] = Field(
        default_factory=lambda: {
            "free": TierLimit(requests_per_minute=20, burst=5, daily_token_budget=50_000),
            "pro": TierLimit(
                requests_per_minute=120, burst=30, daily_token_budget=1_000_000, max_concurrent=8
            ),
            "enterprise": TierLimit(
                requests_per_minute=1_000,
                burst=200,
                daily_token_budget=20_000_000,
                max_concurrent=32,
            ),
        }
    )
    default_tier: str = "free"
    ip_daily_token_budget: int = 200_000
    concurrency_lease_seconds: int = 300


class AuthSettings(BaseModel):
    enabled: bool = False
    algorithm: Literal["HS256", "RS256"] = "HS256"
    secret: SecretStr | None = None
    jwks_url: str | None = None
    public_key: str | None = None
    issuer: str | None = None
    audience: str | None = None
    leeway_seconds: int = 10


class McpServerConfig(BaseModel):
    name: str
    transport: Literal["stdio", "streamable_http"]
    command: list[str] | None = None
    url: str | None = None


class McpSettings(BaseModel):
    servers: list[McpServerConfig] = Field(default_factory=list)
    # HTTP MCP servers must be listed here (SSRF guard). Hostnames only, e.g. "mcp-demo".
    allowed_hosts: list[str] = Field(default_factory=list)
    connect_timeout_seconds: float = 15.0
    required: bool = False  # True: a server that cannot connect fails startup


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    app: AppSettings = Field(default_factory=AppSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    orchestrator: OrchestratorSettings = Field(default_factory=OrchestratorSettings)
    tracing: TracingSettings = Field(default_factory=TracingSettings)
    prompts: PromptStoreSettings = Field(default_factory=PromptStoreSettings)
    evaluation: EvalSettings = Field(default_factory=EvalSettings)
    guardrails: GuardrailSettings = Field(default_factory=GuardrailSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    ratelimit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    tools: ToolsSettings = Field(default_factory=ToolsSettings)
    mcp: McpSettings = Field(default_factory=McpSettings)


@lru_cache
def get_settings() -> Settings:
    return Settings()
