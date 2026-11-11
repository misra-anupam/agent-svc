# Plan: `agent-svc` — Composable Agent-as-a-Microservice Template

## Context
A template repo that wraps an agent (Google ADK / LangGraph / deepagents / MS Agent Framework / CrewAI) in an async FastAPI service. Every concern sits behind a port and is chosen by config. Docker Compose profiles assemble the matching infra. Target: `agent-svc/` (empty, not a git repo; uv 0.11.1 and docker present).

Decisions already made: all 5 orchestrators, every pluggable option wired, all guardrail options, phased delivery with checkpoints.
Execution (per your last message): copy this plan into the repo as `docs/PLAN.md`, `git init` and commit it, then implement phase by phase in auto mode. At each checkpoint the phase's exit criteria must be green before the next phase starts. Each phase ends in its own commit.

## Composability principles (apply to every phase)
1. **Port + Adapter + Registry**. Every swappable concern defines a `Protocol` in `domain/ports/`, adapters in `adapters/<port>/<name>.py`, and a `Registry[Port]` that maps `name → factory(settings)`.
2. **Selection by config only**. `Settings.<component>.provider: Literal[...]` picks the adapter. No `if provider ==` anywhere outside the registry (OCP).
3. **Third-party plugins**. Registries also load Python entry points (`agent_svc.<port>` groups), so an external package can add an adapter without editing this repo.
4. **Optional extras**. Heavy SDKs live in `uv` extras (`[langgraph]`, `[adk]`, `[crewai]`, `[mlflow]`, `[langfuse]`, `[nemo]`, …). The adapter is imported lazily and fails with a clear `ConfigurationError` that names the missing extra.
5. **Decorators for cross-cutting behaviour** (composition over inheritance): `RetryingLLM`, `BreakerLLM`, `FallbackLLM`, `CachedAgent`, `GuardedAgent`, `TracedAgent`, `MeteredAgent`. The container stacks them in a configurable order, and each wraps the same port.
6. **Import boundaries** enforced in CI with `import-linter`: `domain` imports nothing, `application` imports `domain` only, `adapters` import `domain`, and `api` imports `application`.
7. **Reusable standalone modules**: `resilience`, `ratelimit`, `tracing`, `guardrails` and `memory` have no dependency on FastAPI, so they can be lifted into other projects.
8. **Shared contract-test suites**. One suite per port runs against every adapter, so a new adapter is proven by subclassing the suite.
9. **Presets**. `env/*.env` files and compose profiles are named bundles (e.g. `local-min`, `oss-observability`, `enterprise`), and any single choice can be overridden.
10. **Scaffolding**. `make new-adapter PORT=guardrails NAME=foo` generates the adapter, registry entry, contract test and docs stub.

## Repo layout
```
agent-svc/
  pyproject.toml  uv.lock  Makefile  .env.example  .importlinter  .pre-commit-config.yaml
  env/ local-min.env  oss-observability.env  enterprise.env
  docker/ Dockerfile  compose.yaml  otel-collector.yaml  prometheus.yml  loki.yaml  promtail.yaml  grafana/
  mkdocs.yml  docs/ (PLAN.md, architecture, adrs/, how-to/, security.md, ops/)
  src/agent_svc/
    main.py  container.py  registry.py  plugins.py
    config/        # pydantic-settings, one sub-settings class per component
    api/ v1/ v2/   # routers, schemas (pydantic), errors (problem+json), middleware, deps
    domain/        # dataclasses (frozen, slots), exceptions, ports/
    application/   # ChatService, ThreadService, MemoryService, EvalService
    adapters/      # orchestrators llm mcp memory cache ratelimit resilience tracing prompts eval guardrails auth
    observability/ # structlog, OTel metrics, Prometheus
  evals/golden/    # *.jsonl datasets and thresholds.yaml
  tests/ unit/ contract/ integration/ e2e/ golden/ load/
  scripts/ run_golden.py  gen_openapi.py  new_adapter.py  ui/ (static SSE demo UI)
```

## Cross-cutting design rules
- Pydantic is used at the API edge and for settings. Internally there are frozen slotted dataclasses, with explicit mappers at the boundary.
- Everything is async. Blocking SDK calls go through `anyio.to_thread`. Shared clients (httpx, redis, asyncpg) are created once in `lifespan` and closed in reverse order.
- Errors map to HTTP as RFC 9457 `application/problem+json`: 400, 401, 403, 404, 409, 413, 422, 429 (with `Retry-After`), 499, 502, 503 and 504. Mid-stream errors become an SSE `run.error` event.
- Versioning: URL `/v1` and `/v2` (a stub proves the mechanism), plus an agent version registry (`agents/<name>@N`). The agent version is part of every response header, trace attribute and cache key.
- SSE contract: `POST /v1/threads/{id}/chat` emits typed events with `id` and `Last-Event-ID` resume (Redis stream buffer) and a 15s heartbeat. Events are `run.started`, `message.delta`, `step.started|finished` (kind = tool | rag | agent | mcp | guardrail, with `parent_id` for nesting and truncated args/results), `usage`, `message.completed`, `run.error` and `run.completed`. Client disconnect cancels the run.

---
## Phase 0 — Bootstrap (commit the plan)
- Write this plan to `docs/PLAN.md`; `git init`; add `.gitignore`; first commit.
- **Exit**: repo has one commit containing the plan.

## Phase 1 — Foundation and composition kernel
Tasks:
- uv project, Python 3.12, ruff, mypy (strict), pytest-asyncio, import-linter, pre-commit, Makefile targets.
- `config/`: `Settings` composed of `LLMSettings`, `OrchestratorSettings`, `TracingSettings`, `PromptStoreSettings`, `EvalSettings`, `GuardrailSettings`, `MemorySettings`, `CacheSettings`, `RateLimitSettings`, `AuthSettings`, `McpSettings`. Selectors are `Literal` enums, secrets are `SecretStr`, and an `env_nested_delimiter` is set.
- `domain/`: dataclasses (`AgentRequest`, `AgentEvent` union, `Message`, `Usage`, `Principal`, `ToolSpec`), the exception hierarchy, and every port (`AgentRunner`, `LLMClient`, `ToolProvider`, `ThreadMemory`, `UserMemory`, `LongTermMemory`, `Cache`, `RateLimiter`, `TokenBudget`, `Tracer`, `PromptStore`, `Evaluator`, `Guardrail`, `Authenticator`).
- `registry.py`, `plugins.py` (entry-point loading, lazy-import helper, missing-extra error), and `container.py` (builds adapters and stacks decorators, with ordered startup and teardown).
- `main.py`: `create_app()`, lifespan, request-id middleware, problem+json handlers, `/healthz` (liveness) and `/readyz` (checks the container's dependencies), `/metrics`, structlog JSON logging.
- `docker/Dockerfile` (multi-stage uv, non-root, read-only rootfs), a base `compose.yaml` (`api`, `redis`, `postgres`), and CI (lint, types, tests, import boundaries).
- **Exit**: `make check` is green, OpenAPI is served, `/readyz` reflects Redis and Postgres state, an unknown provider or missing extra gives a clear startup error, and an import-linter violation fails CI.

## Phase 2 — Vertical slice: LangGraph + LLM resilience + SSE
Tasks:
- `adapters/llm/litellm.py` plus `FallbackLLM(primary, secondary)` that composes `RetryingLLM` (tenacity: exponential backoff with jitter, retryable errors only, stop budget) and `CircuitBreaker` (async, half-open probes, per-provider state, metrics).
- `adapters/orchestrators/langgraph.py` maps LangGraph stream events to `AgentEvent`. The shared `OrchestratorContractSuite` and a `FakeLLM` are added for deterministic tests.
- `ChatService` and the SSE router, with cancellation and error-event handling.
- Static demo UI (`scripts/ui`) renders step events as nested collapsed `<details>`.
- **Exit**: `curl -N` shows the full event sequence with the fake LLM, killing the primary LLM triggers a visible fallback, the breaker opens after N failures and closes after the cooldown, and the contract suite passes for LangGraph.

## Phase 3 — State, caching and protection
Tasks:
- Memory: thread (Redis, TTL and windowing), user (Postgres), long-term (pgvector with an embedding port and per-tenant namespaces), behind `MemoryFacade`. The app summarises and trims context to a token budget.
- Cache: a `Cache` port with `RedisCache` and an optional semantic cache. The key is hash(normalized prompt, agent version, model, user scope). Includes single-flight stampede protection, `CachedAgent` decorator, and TTL and bypass headers.
- Auth: `Authenticator` with `NoAuth`, `JwtAuth` (HS256/RS256/JWKS, alg pinning, iss/aud/exp, scopes, tier and tenant claims), toggled by `AUTH_ENABLED`.
- Rate limiting: Redis Lua token bucket per tier, with a throttle mode (bounded delay) and a reject mode (429 plus `Retry-After`), keyed by principal or IP. `TokenBudget` per IP (daily and monthly) uses atomic Lua and records actual `usage`.
- Concurrency: a per-instance semaphore, a global Redis cap, 503 backpressure, graceful SIGTERM drain, `Idempotency-Key`, and SSE resume.
- **Exit**: testcontainers integration tests pass for every item above, a locust/k6 script shows tier limits and budget cutoffs, the cache hit rate is visible in metrics, and auth on/off both pass the e2e tests.

## Phase 4 — Observability
Tasks:
- `Tracer` adapters: `otel` (OTLP), `mlflow` (tracing), `langfuse`. A `TracedAgent` decorator creates run, LLM, tool and step spans with agent version, tier, token and cost attributes.
- Metrics: request, token, cache, limiter, breaker and fallback counters. Logs are JSON with a trace id.
- Compose profiles: `otel` (collector, Tempo or Jaeger), `lgtm` (Prometheus, Grafana, Loki, Promtail with provisioned dashboards), `mlflow` and `langfuse`.
- **Exit**: one request is visible in each backend and the Grafana dashboards render real data. Switching tracers needs only an env change.

## Phase 5 — Remaining orchestrators and MCP
Tasks:
- Adapters for Google ADK, deepagents, MS Agent Framework and CrewAI. Each lives in its own extra, maps native events to `AgentEvent`, and passes `OrchestratorContractSuite`. Versions are pinned after checking current docs.
- MCP: `ToolProvider` adapters (stdio and streamable-http) with an allowlist and SSRF guard. A per-orchestrator `ToolTranslator` converts MCP tools into native tool objects. `mcp-demo` profile.
- CI matrix tests each extra in an isolated env (conflict handling in uv).
- **Exit**: the same golden prompt yields the same event-shape sequence on all five orchestrators, and MCP tool calls appear as `step` events on each.

## Phase 6 — Guardrails
Tasks:
- `GuardrailChain` composite (input rails then output rails, with block, redact or warn actions and streaming-aware output checks).
- Adapters: `builtin` (injection heuristics, secrets, length), `presidio` (PII), `guardrails_ai` (Hub validators), `nemo` (Colang, telemetry off), `llama_guard`, `prompt_guard`, and `llm_guard` (deprecated, flagged as archived upstream).
- Heavy models run as `guardrails` and `guardrails-gpu` profile sidecars. Per-tier policy selects which rails run.
- **Exit**: an adversarial suite (OWASP LLM01/02/05/07) passes per adapter, blocked requests return 422 or an SSE `run.error` with a policy code, and the guardrail latency overhead is measured and documented.

## Phase 7 — Prompts, evaluation and golden dataset
Tasks:
- `PromptStore` adapters: `mlflow`, `langsmith`, and a `file` fallback. Prompts are versioned and aliased, and cached with TTL.
- `Evaluator` adapters: `mlflow_genai` (`mlflow.genai.evaluate`), `langsmith` (evals), and `adk_eval`. All are driven by one `GoldenDatasetRunner` over `evals/golden/*.jsonl` with `thresholds.yaml`.
- `make eval` and a CI gate that fails on regression. The agent version is recorded with every run.
- **Exit**: the golden run executes on all three evaluators and fails correctly when a threshold is violated.

## Phase 8 — Security, docs and hardening
Tasks:
- `docs/security.md` maps both the OWASP Web Top 10 and the OWASP LLM Top 10 (2025) to concrete controls and tests, and I'll check the current editions on owasp.org when writing it. Add security headers, strict CORS, pip-audit, Trivy and bandit in CI, and an SBOM.
- mkdocs-material site: architecture, ADRs, how-to-add-an-adapter, presets, ops runbook, API reference from OpenAPI. A `docs` profile serves it.
- Full compose smoke test per preset, and a load and soak test.
- **Exit**: `mkdocs build --strict` is clean, all CI jobs are green, every compose preset boots, and the security mapping has a test or doc link for each row.

## Risks
- Framework APIs drift, so I verify the current docs at the start of each phase and pin versions.
- The frameworks' dependencies may conflict, which uv `conflicts` declarations and per-extra CI environments should handle.
- NeMo and Llama Guard are heavy, so they are sidecars and skipped in default CI.
- Guardrail research sources are vendor blogs, so I'll verify against upstream repos. LLM Guard is reportedly archived.

## Verification (every phase)
`uv run ruff check && uv run mypy && uv run lint-imports && uv run pytest`, plus the phase's exit criteria, a compose smoke test for the touched profiles, and a git commit per phase.

---
## Progress log
- **Phase 0–2**: done (foundation, LangGraph slice, resilient LLM stack, SSE).
- **Phase 3**: done. Memory (thread/user/long-term), exact-match answer cache with single-flight,
  JWT (switchable), tier rate limits, IP + user token budgets, concurrency leases, Idempotency-Key,
  SSE resume via RunBuffer. Verified with fakeredis (Lua), real pgvector (`make test-integration`)
  and a live load run (`tests/load/limits_smoke.py`).
  - **Deferred**: the *semantic* answer cache (`CacheSettings.semantic` is reserved, not implemented).
    Exact-match caching is shipped. Plan: implement on top of `Embedder` + pgvector in Phase 8.
  - **Not yet run**: full `docker compose up` smoke with Redis and Postgres (done in Phase 8).
- **Phase 4**: done. `Tracer` port with `none | otel | mlflow | langfuse` adapters and a backend-neutral
  `TracedRunner` (run -> step -> llm span tree, token/cost attributes, prompts/answers captured only when
  `capture_content=true`). Compose profiles `otel` (collector + Jaeger), `lgtm` (Prometheus, Loki, Promtail,
  Grafana with a provisioned dashboard), `mlflow`, `langfuse`. **Verified live**: one request produced a
  trace in Jaeger, MLflow and Langfuse; Prometheus scraped metrics; Loki received JSON logs; Grafana
  provisioned the dashboard.
  - Environment notes: Docker Hub's `minio/minio` is gone, so the Langfuse profile uses Chainguard's MinIO
    with a busybox init job (volume ownership 65532). MLflow is published on host port 5001 because macOS
    AirPlay owns 5000. Compose here is the standalone `docker-compose` binary (Makefile auto-detects).
  - Not verified visually: the Grafana panels were provisioned and queried through the datasource proxy,
    but I did not open the dashboard in a browser.
- **Phase 5**: done. Tool calling added to the `LLMClient` port; `ToolProvider` port with local (calculator,
  current_time), MCP (stdio + streamable-http, SSRF allowlist, namespaced `server__tool`) and a guard layer
  (allowlist, jsonschema arg validation, timeout, output cap). Adapters for LangGraph, deepagents, ADK, MAF
  and CrewAI each bridge to the resilient LLM port, so retry/breaker/fallback/metering apply uniformly. One
  orchestrator-agnostic contract suite (5 scenarios x 5 providers) passes: plain-answer grammar, nested
  balanced tool steps, failing tool then recovery, invalid-args rejection, runaway-loop -> `iteration_limit`.
  All extras resolve in one lock (MAF uses `agent-framework-core`; the meta-package has conflicting pins).
  CI has an `orchestrator-isolation` matrix (core + one extra each). `mcp-demo` compose profile.
  - Framework quirks handled: MAF silently stops at max iterations (cap enforced in the runner); CrewAI
    runs sync in a worker thread (bridged to the service loop; iteration cap enforced in the LLM bridge).
  - Not run: the isolation matrix itself (needs CI); `mcp-demo` was validated with `compose config`, not booted.
