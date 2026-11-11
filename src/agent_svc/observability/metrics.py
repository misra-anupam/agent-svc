"""Application metrics as OpenTelemetry instruments (API only).

Instruments bind to whatever ``MeterProvider`` is installed globally; until one is (see
``adapters/tracing/otel_metrics.py``) they are no-ops, so importing this module costs nothing.
"""

from __future__ import annotations

from opentelemetry import metrics

_meter = metrics.get_meter("agent-svc")

HTTP_REQUESTS = _meter.create_counter(
    "agent.http.requests", unit="{request}", description="HTTP requests"
)
HTTP_LATENCY = _meter.create_histogram(
    "agent.http.request.duration",
    unit="s",
    description="HTTP request latency",
    explicit_bucket_boundaries_advisory=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60],
)
HTTP_INFLIGHT = _meter.create_up_down_counter(
    "agent.http.inflight_requests", unit="{request}", description="In-flight HTTP requests"
)

LLM_CALLS = _meter.create_counter("agent.llm.calls", unit="{call}", description="LLM calls")
LLM_TOKENS = _meter.create_counter("agent.llm.tokens", unit="{token}", description="LLM tokens")
LLM_RETRIES = _meter.create_counter(
    "agent.llm.retries", unit="{retry}", description="LLM retry attempts"
)
LLM_FALLBACKS = _meter.create_counter(
    "agent.llm.fallbacks", unit="{fallback}", description="Fallbacks to the secondary LLM"
)
BREAKER_STATE = _meter.create_gauge(
    "agent.breaker.state", description="Circuit breaker state: 0=closed 1=half_open 2=open"
)
BREAKER_TRANSITIONS = _meter.create_counter(
    "agent.breaker.transitions", unit="{transition}", description="Circuit breaker transitions"
)

CACHE_REQUESTS = _meter.create_counter(
    "agent.cache.requests", unit="{lookup}", description="Answer cache lookups"
)
RATE_LIMITED = _meter.create_counter(
    "agent.ratelimit.rejections", unit="{request}", description="Rejected or throttled requests"
)

GUARDRAIL_CHECKS = _meter.create_counter(
    "agent.guardrail.checks", unit="{check}", description="Guardrail verdicts"
)
GUARDRAIL_LATENCY = _meter.create_histogram(
    "agent.guardrail.duration",
    unit="s",
    description="Guardrail latency per rail",
    explicit_bucket_boundaries_advisory=[0.001, 0.005, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5],
)
