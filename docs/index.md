# agent-svc

A template for running an AI agent as a production microservice: async FastAPI, SSE streaming, and
every concern behind a port you can swap by configuration.

| Concern | Choices |
|---|---|
| Orchestrator | LangGraph, deepagents, Google ADK, Microsoft Agent Framework, CrewAI (`echo` for demos) |
| LLM | LiteLLM with retry, circuit breaker and primary/secondary fallback |
| Tools | local tools, MCP servers (stdio / streamable-http) |
| State | thread (Redis), user (Postgres), long-term (pgvector) memory |
| Protection | tiered rate limits, token budgets, concurrency caps, JWT, answer cache (exact + optional semantic) |
| Guardrails | builtin, Presidio, Guardrails AI, Prompt Guard, Llama Guard, NeMo, LLM Guard (deprecated) |
| Observability | OpenTelemetry, MLflow, Langfuse, Prometheus, Grafana, Loki |
| Prompts / evals | file, MLflow, LangSmith stores; local, MLflow GenAI, LangSmith, ADK evaluators |

```bash
make install && make check
AGENT_ORCHESTRATOR__PROVIDER=echo make run     # http://localhost:8000/ui
curl -N -X POST localhost:8000/v1/threads/demo/chat \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}]}'
```

!!! note "Not implemented"
    The plan's named, versioned **agent registry** (`agents/<name>@N`, `/v2` stub) was deliberately
    left out. Today there is one agent identified by `app.agent_name` / `app.agent_version`
    (see [Project plan & status](PLAN.md)).
