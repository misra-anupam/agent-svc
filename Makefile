COMPOSE ?= $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo docker-compose)
PROFILES ?=
.PHONY: test-integration install check lint types arch test run up docs
install:
	uv sync --group dev --extra litellm --extra langgraph --extra otel --extra langfuse --extra mlflow
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
# Golden-dataset gate: scripted model through the real stack, no API key. Fails on any threshold.
eval:
	MLFLOW_DISABLE_AGENT_HINT=1 uv run agent-svc-eval $(EVAL_ARGS)
run:
	uv run uvicorn agent_svc.main:app --reload --factory
up:
	$(COMPOSE) -f docker/compose.yaml $(PROFILES) up --build
docs:
	uv run mkdocs serve
