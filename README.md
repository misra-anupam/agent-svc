# agent-svc

Composable, async agent-as-a-microservice template. Every concern (orchestrator, LLM, tracing,
prompt store, evaluation, guardrails, memory, cache, auth, MCP) is a port with swappable adapters
selected by config. Docs: `make docs` (serve) or `make docs-build`; security mapping in [`docs/security.md`](docs/security.md);
roadmap and status in [`docs/PLAN.md`](docs/PLAN.md).

```bash
make install   # uv sync
make check     # ruff, mypy, import-linter, pytest
make run       # uvicorn --reload
make up        # docker compose (api, redis, postgres)
make eval      # golden-dataset gate (no API key)
make new-adapter PORT=guardrails NAME=acme
```

Select components with env vars, e.g. `AGENT_ORCHESTRATOR__PROVIDER=langgraph`; install heavy SDKs
as extras: `uv sync --extra langgraph`.
