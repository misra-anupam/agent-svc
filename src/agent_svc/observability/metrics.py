from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter("agent_http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LATENCY = Histogram(
    "agent_http_request_seconds",
    "HTTP request latency",
    ["method", "route"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
HTTP_INFLIGHT = Gauge("agent_http_inflight_requests", "In-flight HTTP requests")

LLM_CALLS = Counter("agent_llm_calls_total", "LLM calls", ["model", "outcome"])
LLM_TOKENS = Counter("agent_llm_tokens_total", "LLM tokens", ["model", "direction"])
LLM_RETRIES = Counter("agent_llm_retries_total", "LLM retry attempts", ["provider"])
LLM_FALLBACKS = Counter("agent_llm_fallbacks_total", "Fallbacks to secondary LLM", ["from_model"])
BREAKER_STATE = Gauge("agent_breaker_state", "0=closed 1=half_open 2=open", ["name"])
BREAKER_TRANSITIONS = Counter(
    "agent_breaker_transitions_total", "Breaker transitions", ["name", "to"]
)

CACHE_REQUESTS = Counter("agent_cache_requests_total", "Answer cache lookups", ["result"])
RATE_LIMITED = Counter(
    "agent_rate_limited_total", "Rejected/throttled requests", ["reason", "tier"]
)

GUARDRAIL_CHECKS = Counter(
    "agent_guardrail_checks_total", "Guardrail verdicts", ["rail", "stage", "action"]
)
GUARDRAIL_LATENCY = Histogram(
    "agent_guardrail_seconds",
    "Guardrail latency per rail",
    ["rail", "stage"],
    buckets=(0.001, 0.005, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)
