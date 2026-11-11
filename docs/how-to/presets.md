# Configuration and presets

Settings are read from environment variables (and `.env`) with the `AGENT_` prefix and `__` for
nesting: `AGENT_LLM__PRIMARY_MODEL`, `AGENT_GUARDRAILS__FAIL_MODE`. Invalid values fail at startup.
`.env.example` lists the common ones.

## Env presets (`env/`)

| File | Use |
|---|---|
| `local-min.env` | Echo orchestrator, no auth, no external services beyond optional Redis |
| `enterprise.env` | JWT, Redis limits/cache/memory, Postgres + pgvector, detached runs |
| `oss-observability.env` | LangGraph + OpenTelemetry to the bundled collector |

Any single setting can still be overridden on top of a preset.

## Compose profiles (`docker/compose.yaml`)

`api`, `redis` and `postgres` always start. Add profiles with `--profile`:

| Profile | Adds | Notes |
|---|---|---|
| `otel` | OpenTelemetry Collector + Jaeger (`:16686`) | |
| `lgtm` | Prometheus, Loki, Promtail, Grafana (`:3000`) with a provisioned dashboard | |
| `mlflow` | MLflow tracking server (host port `5001`) | |
| `langfuse` | Langfuse + its Postgres, ClickHouse, MinIO | |
| `mcp-demo` | a FastMCP server on `:8080` | set `AGENT_MCP__*` as in the compose comments |
| `guardrails` | Prompt Guard (TEI) and NeMo Guardrails server | gated model: needs `HF_TOKEN` |
| `guardrails-gpu` | Llama Guard via vLLM | needs an NVIDIA GPU and `HF_TOKEN` |

```bash
AGENT_EXTRAS="--extra litellm --extra langgraph --extra otel" \
AGENT_TRACING__PROVIDER=otel \
docker compose -f docker/compose.yaml --profile otel --profile lgtm up --build
```

`AGENT_EXTRAS` chooses which uv extras are installed into the image (the default keeps it small).

!!! warning "Verification status"
    Verified live: `otel`, `lgtm`, `mlflow`, `langfuse` (Phase 4). `mcp-demo`, `guardrails` and
    `guardrails-gpu` pass `docker compose config` but were not booted. The sidecar images, model ids and
    the NeMo request shape are version-dependent: check them against your versions.
