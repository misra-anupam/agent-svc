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
