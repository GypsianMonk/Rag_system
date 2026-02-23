.PHONY: help install dev test lint format typecheck clean docker-up docker-down migrate

PYTHON := python3
PIP    := pip3
APP    := app

help:
	@echo "Enterprise RAG System — available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Setup ──────────────────────────────────────────────────────────────────────
install: ## Install all dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt

env: ## Copy .env.example → .env
	@test -f .env || cp .env.example .env && echo "Created .env — fill in your secrets"

# ── Development ────────────────────────────────────────────────────────────────
dev: ## Run dev server with hot reload
	uvicorn $(APP).main:app --host 0.0.0.0 --port 8000 --reload

# ── Quality ────────────────────────────────────────────────────────────────────
lint: ## Run ruff linter
	ruff check $(APP) tests

format: ## Auto-format with ruff + isort
	ruff format $(APP) tests
	isort $(APP) tests

typecheck: ## Type-check with mypy
	mypy $(APP) --ignore-missing-imports

# ── Tests ──────────────────────────────────────────────────────────────────────
test: ## Run full test suite
	pytest tests/ -v --cov=$(APP) --cov-report=term-missing --cov-report=html

test-unit: ## Run unit tests only
	pytest tests/unit/ -v -x

test-integration: ## Run integration tests (requires infra)
	pytest tests/integration/ -v -x

# ── Database ───────────────────────────────────────────────────────────────────
migrate: ## Run Alembic migrations
	alembic upgrade head

migrate-create: ## Create new migration (usage: make migrate-create MSG="add column")
	alembic revision --autogenerate -m "$(MSG)"

migrate-downgrade: ## Downgrade one revision
	alembic downgrade -1

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build: ## Build Docker image
	docker build -f docker/Dockerfile --target production -t rag-system:latest .

docker-up: ## Start all services
	docker compose up -d

docker-down: ## Stop all services
	docker compose down

docker-logs: ## Tail API logs
	docker compose logs -f api

docker-shell: ## Open shell in API container
	docker compose exec api bash

# ── Evaluation ────────────────────────────────────────────────────────────────
eval: ## Run evaluation suite
	$(PYTHON) scripts/run_evaluation.py

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean: ## Remove cache and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov .mypy_cache .ruff_cache .pytest_cache dist build
