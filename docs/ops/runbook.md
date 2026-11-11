# Runbook

## Health

| Endpoint | Meaning |
|---|---|
| `/healthz` | process is up (liveness) |
| `/readyz` | required dependencies reachable; 503 with per-check detail otherwise |
| `/metrics` | Prometheus |

Optional dependencies (Redis, Postgres, MCP servers) that fail to start are logged and the service
runs degraded unless marked `required`.

## Key metrics and what to do

| Metric | Symptom | Action |
|---|---|---|
| `agent_breaker_state` = 2, `agent_llm_fallbacks_total` rising | primary model failing | check provider status; fallback is serving; consider raising `llm.breaker_reset_seconds` |
| `agent_rate_limited_total` | clients throttled | confirm it is expected; tune `ratelimit.tiers` |
| `agent_guardrail_checks_total{action="block"}` spike | attack or a too-eager rail | sample the `policy_violation` events; check `agent_guardrail_seconds` |
| `agent_guardrail_checks_total{action="block"}` with reason `rail_error:*` | a rail/sidecar is down and failing closed | restore the sidecar; do **not** switch to `open` casually |
| `agent_cache_requests_total{result="semantic_hit"}` | semantic cache serving | disable `cache.semantic` if answers look wrong |
| `agent_http_inflight_requests` near the concurrency cap | back-pressure (503) | scale out; limits are shared through Redis |

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
* **Presidio `spaCy model ... is missing`**: run `python -m spacy download en_core_web_sm` in the image.
* **MCP server unreachable**: allowed-host mismatch or the server is down; HTTP MCP hosts must be listed in
  `mcp.allowed_hosts`.

## Evaluating a release

Run `make eval` (and `--real` on a schedule). Compare the report's `agent_version` and `prompt` with the
previous run; bump `app.agent_version` when behaviour changes so cache entries are not reused.
