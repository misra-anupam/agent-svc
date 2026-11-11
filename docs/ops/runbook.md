# Runbook

## Health

| Endpoint | Meaning |
|---|---|
| `/healthz` | process is up (liveness) |
| `/readyz` | required dependencies reachable; 503 with per-check detail otherwise |

Optional dependencies (Redis, Postgres, MCP servers) that fail to start are logged and the service
runs degraded unless marked `required`.

## Key metrics and what to do

Metrics are OpenTelemetry instruments exported over OTLP to `tracing.otlp_endpoint` every
`metrics.export_interval_seconds` (default 15 s); the bundled collector only logs them, so point it at your
metrics backend. Turn them off with `AGENT_METRICS__ENABLED=false`.

| Metric | Symptom | Action |
|---|---|---|
| `agent.breaker.state` = 2, `agent.llm.fallbacks` rising | primary model failing | check provider status; fallback is serving; consider raising `llm.breaker_reset_seconds` |
| `agent.ratelimit.rejections` | clients throttled | confirm it is expected; tune `ratelimit.tiers` |
| `agent.guardrail.checks{action="block"}` spike | attack or a too-eager rail | sample the `policy_violation` events; check `agent.guardrail.duration` |
| `agent.guardrail.checks{action="block"}` with reason `rail_error:*` | a rail is raising and failing closed | fix the rail; do **not** switch to `open` casually |
| `agent.cache.requests{result="semantic_hit"}` | semantic cache serving | disable `cache.semantic` if answers look wrong |
| `agent.http.inflight_requests` near the concurrency cap | back-pressure (503) | scale out; limits are shared through Redis |

## Deploy and shutdown

On SIGTERM the service stops accepting runs and drains in-flight ones for up to
`app.shutdown_grace_seconds` (default 20 s). Set your orchestrator's termination grace period above it.
With `app.detach_on_disconnect=true` and the Redis run buffer, clients can resume a run on another instance.

## Common failures

* **Startup `ConfigurationError ... uv sync --extra X`**: that provider's extra is not installed in the image;
  rebuild with `AGENT_EXTRAS`.
* **Startup `unsafe production configuration`**: production requires auth, explicit CORS origins and
  fail-closed guardrails.
* **`iteration_limit` errors**: the agent looped past `orchestrator.max_iterations`; inspect the trace for a
  tool that keeps failing.
* **MCP server unreachable**: allowed-host mismatch or the server is down; HTTP MCP hosts must be listed in
  `mcp.allowed_hosts`.

## Evaluating a release

Run `make eval` (and `--real` on a schedule). Compare the report's `agent_version` and `prompt` with the
previous run; bump `app.agent_version` when behaviour changes so cache entries are not reused.
