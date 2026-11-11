# Security

How `agent-svc` maps to the **OWASP Top 10:2025** (web) and the **OWASP Top 10 for LLM Applications 2025**.
Editions checked against the primary sources on 2026-11-11:
[owasp.org Top 10:2025](https://top10.owasp.org/2025) and
[genai.owasp.org LLM Top 10](https://genai.owasp.org/llm-top-10/).

Status: **Covered** = a control exists and a test exercises it. **Partial** = a control exists with a
stated gap. **Out of scope** = belongs to the deployer/platform; what we do and don't do is stated.
Paths are relative to the repo root.

!!! warning "What this is not"
    A template ships defaults, not a certification. Nothing here replaces a threat model and a
    pen-test of *your* deployment (your tools, your data, your model).

## OWASP Top 10:2025 (web)

| ID | Category | Status | Controls | Evidence |
|---|---|---|---|---|
| A01 | Broken Access Control (includes SSRF) | Covered | JWT auth switchable per environment; scopes/tier/tenant claims. Thread and memory keys are `tenant/subject/thread`, so another user's thread is a 404 (no IDOR). MCP: hostname allow-list, cloud-metadata hosts always blocked, redirects disabled. | `tests/e2e/test_phase3_protection.py` (IDOR), `tests/unit/test_memory.py` (tenant isolation), `tests/unit/test_mcp.py` (SSRF), `adapters/auth/jwt.py` |
| A02 | Security Misconfiguration | Covered | Production refuses to start with auth off, `*` CORS origins or `guardrails.fail_mode=open`. Docs/OpenAPI/demo UI are disabled in production. Strict CORS (listed origins; GET/POST/DELETE only). Hardening headers (CSP, HSTS in production, nosniff, frame deny, COOP/CORP, Permissions-Policy). Container: non-root, read-only rootfs, `no-new-privileges`. | `tests/e2e/test_security_hardening.py`, `config/settings.py` (`_production_must_be_safe`), `docker/compose.yaml` |
| A03 | Software Supply Chain Failures | Partial | Single `uv.lock` (hashes), CI runs `bandit`, `pip-audit` (core runtime dependencies are a hard gate; every extra is informational) and Trivy secret/misconfig scanning, and publishes a CycloneDX SBOM. Extras are pinned by lock, and the `orchestrator-isolation` CI matrix proves each extra works alone. **Known findings (2026-11-11, `pip-audit`):** core runtime dependencies: none. Optional extras: `filelock` 3.32.7 (13 advisories, fixed in 4.x; pulled in by `litellm`/`huggingface-hub` and held back on Python < 3.15 by the resolver) and `chromadb` 1.5.9 (4 advisories, no fix released; pulled in only by the `crewai` extra). Review before enabling those extras in production. Gap: the lock is only as safe as its upstreams (e.g. `llm-guard` is archived upstream and is not installed by default); there is no signature verification of dependencies or images. | `.github/workflows/ci.yml`, `uv.lock` |
| A04 | Cryptographic Failures | Partial | JWT algorithm is pinned (a token with `alg=none` or a different algorithm is rejected), `iss`/`aud`/`exp` checked with bounded leeway. Secrets are `SecretStr` and never logged. Gap: TLS is **not** terminated by this service: run it behind a TLS proxy (HSTS is sent in production). Redis/Postgres connections are plaintext in the sample compose. | `tests/unit/test_auth.py` |
| A05 | Injection | Covered | Request bodies are validated (pydantic; size and count limits). SQL uses bound parameters (asyncpg). Tool arguments are JSON-Schema-validated before execution. The calculator tool evaluates an AST allow-list (no `eval`). Prompt injection: see LLM01. | `tests/unit/test_mcp.py`, `adapters/tools/local.py`, `tests/contract/test_orchestrator_contract.py` (invalid args rejected) |
| A06 | Insecure Design | Partial | Layered architecture enforced by import-linter; rate limits, token budgets, concurrency caps and iteration caps are on the design's critical path, not bolt-ons. Guardrails default to fail-closed. Gap: no formal threat model is shipped; write one for your tools. | `.importlinter`, `tests/unit/test_admission.py` |
| A07 | Authentication Failures | Partial | Pinned algorithm, expiry enforced, per-IP and per-user limits throttle credential stuffing against the token endpoint of *your* IdP (this service only validates tokens). Gap: no token revocation/denylist; keep lifetimes short or put an introspecting gateway in front. | `tests/unit/test_auth.py`, `tests/load/limits_smoke.py` |
| A08 | Software or Data Integrity Failures | Partial | Golden-dataset gate runs in CI; prompts are versioned and aliased (file store lives in git); answer cache is partitioned by agent version/scope. Gap: cached answers are not signed; whoever can write Redis can poison the cache, so isolate and authenticate Redis. | `evals/golden/`, `adapters/cache/cached_runner.py` |
| A09 | Security Logging and Alerting Failures | Partial | Structured JSON logs with request id and trace id; guardrail verdicts, rate-limit rejections and breaker transitions are Prometheus counters; every run is traceable. Prompts/answers are only captured when `capture_content=true`. Gap: no shipped alert rules (add alerts on `agent_guardrail_checks_total{action="block"}`, `agent_rate_limited_total`). | `observability/metrics.py`, `docker/grafana/` |
| A10 | Mishandling of Exceptional Conditions | Covered | One error model (RFC 9457 problem+json) with no stack traces; mid-stream failures become an SSE `run.error`; a broken guardrail fails closed; cache/embedder outages degrade to normal answering; breaker + fallback for the LLM; client disconnect cancels the run. | `tests/e2e/test_chat_sse.py`, `tests/unit/test_llm_resilience.py`, `tests/unit/test_guarded_runner.py` |

## OWASP Top 10 for LLM Applications 2025

| ID | Category | Status | Controls | Evidence |
|---|---|---|---|---|
| LLM01 | Prompt Injection | Partial | Input guardrails on the newest user message: normalised heuristics (builtin) plus optional classifiers (Prompt Guard, Llama Guard, NeMo). Tools are allow-listed, argument-validated and capped, so an injected instruction has limited reach. **Gap:** only the newest user message is checked; text that enters via history, memory, retrieved documents or *tool output* is trusted. Indirect injection is therefore only mitigated by the tool allow-list and output rails, not detected. Heuristics are bypassable; use a classifier for real coverage. | `tests/unit/test_guardrails_builtin.py` (`tests/guardrails/adversarial.jsonl`), `tests/e2e/test_guardrails_api.py`, golden cases `inject-*` |
| LLM02 | Sensitive Information Disclosure | Partial | Secrets (AWS, GitHub, JWT, private keys, API keys) are redacted on input and output; optional Presidio PII redaction/blocking; answers and prompts are not traced unless opted in; cache is scoped per user by default; thread/memory keys are tenant-isolated. Gap: PII detection is statistical (misses and false positives); the model can still reveal what it was given. | `tests/unit/test_guardrail_adapters.py` (Presidio), golden case `secret-in-answer-is-redacted` |
| LLM03 | Supply Chain | Partial | See A03. Models are called through LiteLLM with explicit model ids; sidecar models (Prompt Guard, Llama Guard) are gated downloads you pin yourself. Gap: model provenance is the deployer's job. | `uv.lock`, `docker/compose.yaml` |
| LLM04 | Data and Model Poisoning | Out of scope (partial) | The service trains nothing. Long-term memory writes can be disabled (`memory.long_term_write=false`); memory is per tenant and user so one user cannot poison another's recall. Gap: no content screening of what is written to memory. | `tests/unit/test_memory.py` |
| LLM05 | Improper Output Handling | Covered (for the API) | Output rails strip `<script>`, event handlers, `javascript:` URLs and markdown images that exfiltrate through query strings. The API returns JSON/SSE with `Content-Type` set and a `default-src 'none'` CSP. Gap: **your front end** must still escape model text; treat it as untrusted. | `tests/guardrails/adversarial.jsonl` (`out-*`), `tests/unit/test_guarded_runner.py` |
| LLM06 | Excessive Agency | Covered (by config) | Tools are opt-in: local tools by name, MCP servers by explicit config; `tools.allowlist` (glob) filters at call time; per-call timeout, output cap and a hard iteration cap (`orchestrator.max_iterations`, error `iteration_limit`). MCP tool names are namespaced `server__tool`. Gap: there is no human-approval step; do not wire destructive tools without adding one. | `tests/contract/test_orchestrator_contract.py` (runaway loop), `tests/unit/test_mcp.py` |
| LLM07 | System Prompt Leakage | Partial | Builtin output rail blocks answers that contain a long verbatim run of the system prompt; input rail blocks "reveal your system prompt" phrasing. Do not put secrets in prompts: paraphrased leakage is not detected. | `tests/guardrails/adversarial.jsonl` (`leak-*`), golden case `leak-system-prompt` |
| LLM08 | Vector and Embedding Weaknesses | Partial | Long-term memory (pgvector) is partitioned per tenant and user. The semantic cache partitions by agent version, scope and earlier conversation. Gap: no embedding-inversion defence; embeddings of sensitive text are sensitive. | `tests/integration/test_postgres_memory.py`, `tests/unit/test_semantic_cache.py` |
| LLM09 | Misinformation | Partial | Golden-dataset gate (answer, tool-trajectory and error expectations) catches regressions; optional MLflow LLM judge. The semantic cache is **off by default** because "similar" is not "identical" and a wrong cached answer repeats. Gap: no hallucination detection at runtime. | `evals/golden/`, `tests/unit/test_golden_eval.py` |
| LLM10 | Unbounded Consumption | Covered | Tiered request rate limits, per-IP and per-user daily token budgets, concurrency leases with 503 back-pressure, max message size/count, iteration cap, per-call timeouts, circuit breaker, graceful drain. Needs Redis for limits to be shared across instances. | `tests/unit/test_admission.py`, `tests/e2e/test_phase3_protection.py`, `tests/load/limits_smoke.py` |

## Operating it safely

1. Run behind a TLS-terminating proxy; set `app.trusted_proxy_hops` to the real proxy count so
   per-IP limits use the client address rather than the proxy's.
2. Production: `AGENT_APP__ENVIRONMENT=production`, `AGENT_AUTH__ENABLED=true`, explicit CORS origins.
   Startup fails otherwise.
3. Keep the guardrail `fail_mode` at `closed`. Add a classifier rail for injection (`prompt_guard`)
   and `presidio` for PII where it matters.
4. Do not enable the semantic cache for answers that must be exact, and keep `cache.scope=user`.
5. Treat all tool output and retrieved text as untrusted input to the model.
6. Isolate Redis and Postgres on a private network with authentication and TLS.

## Reporting a vulnerability

Open a private security advisory on the repository rather than a public issue.
