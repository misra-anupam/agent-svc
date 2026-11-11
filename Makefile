.PHONY: install check lint types arch test run up docs
install:
	uv sync --group dev --extra litellm --extra langgraph
lint:
	uv run ruff check . && uv run ruff format --check .
types:
	uv run mypy
arch:
	uv run lint-imports
test:
	uv run pytest -q
check: lint types arch test
run:
	uv run uvicorn agent_svc.main:app --reload --factory
up:
	docker compose -f docker/compose.yaml up --build
docs:
	uv run mkdocs serve
