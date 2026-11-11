# Configuration and presets

Settings are read from environment variables (and `.env`) with the `AGENT_` prefix and `__` for
nesting: `AGENT_LLM__PRIMARY_MODEL`, `AGENT_GUARDRAILS__FAIL_MODE`. Invalid values fail at startup.
`.env.example` lists the common ones.

## Env presets (`env/`)

| File | Use |
|---|---|
| `local-min.env` | LangGraph + OpenRouter, no auth; needs Postgres for user memory (and Redis, optionally) |
| `enterprise.env` | JWT, Redis limits/cache/memory, Postgres + pgvector, detached runs |

Any single setting can still be overridden on top of a preset.

## Compose profiles (`docker/compose.yaml`)

`api`, `redis`, `postgres`, `otel-collector` + `jaeger` (traces UI on `:16686`) and `mlflow` (prompts and evals, host port `5001`) always start. Add profiles with `--profile`:

| Profile | Adds | Notes |
|---|---|---|
| `mcp-demo` | a FastMCP server on `:8080` | set `AGENT_MCP__*` as in the compose comments |

```bash
docker compose -f docker/compose.yaml up --build
```

`AGENT_EXTRAS` chooses which uv extras are installed into the image (the default keeps it small).

!!! warning "Verification status"
    The `otel-collector`, `jaeger` and `mlflow` services were verified live in Phase 4 but not since the observability changes; re-check them. `mcp-demo` passes `docker compose config` but was not booted.
