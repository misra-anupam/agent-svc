COMPOSE ?= $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo docker-compose)
PROFILES ?=
.PHONY: test-integration install check lint types arch test run up docs docs-build docs-api new-adapter eval
install:
	uv sync --group dev --extra litellm --extra langgraph --extra otel --extra mlflow --extra mlflow_genai
lint:
	uv run ruff check . && uv run ruff format --check .
types:
	uv run mypy
arch:
	uv run lint-imports
test:
	uv run pytest -q
test-integration:
	uv run pytest -q -m integration
check: lint types arch test
# Golden-dataset gate: your configured model (OPENROUTER_API_KEY) through the real stack. Fails on any threshold.
eval:
	MLFLOW_DISABLE_AGENT_HINT=1 uv run agent-svc-eval $(EVAL_ARGS)
run:
	uv run uvicorn agent_svc.main:app --reload --factory
up:
	$(COMPOSE) -f docker/compose.yaml $(PROFILES) up --build
docs:
	uv run mkdocs serve
docs-api:
	uv run python scripts/gen_openapi.py
docs-build: docs-api
	uv run mkdocs build --strict
new-adapter:
	uv run python scripts/new_adapter.py $(PORT) $(NAME)
