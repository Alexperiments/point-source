.PHONY: help install install-dev run run-dev format lint lint-fix test test-cov clean docker-up docker-down docker-logs docker-dev-up docker-dev-down docker-dev-logs docker-dev-restart docker-rebuild-frontend docker-rebuild-backend migration-create migration-upgrade migration-downgrade migration-current migration-history pre-commit-install pre-commit-run createsuperuser

# Default target
.DEFAULT_GOAL := help

# Monorepo directories
BACKEND_DIR := backend
INFRA_DIR := infra
DEVELOPMENT_ENV := $(BACKEND_DIR)/.env.development
PRODUCTION_ENV := $(BACKEND_DIR)/.env.production
INFRA_DEV_COMPOSE := $(INFRA_DIR)/docker-compose.dev.yml
INFRA_COMPOSE := $(INFRA_DIR)/docker-compose.yml

# Colors for help output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Available commands:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

install: ## Install project dependencies
	cd $(BACKEND_DIR) && uv sync

install-dev: ## Install project dependencies including dev dependencies
	cd $(BACKEND_DIR) && uv sync --all-groups

run: ## Run the application in production mode
	cd $(BACKEND_DIR) && uv run uvicorn src.main:app --host 0.0.0.0 --port 8000

run-dev: ## Run the application in development mode with auto-reload
	cd $(BACKEND_DIR) && ENVIRONMENT=development uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir src

format: ## Format code using Ruff
	cd $(BACKEND_DIR) && env -u FORCE_COLOR uv run nox -s fmt

test: ## Run tests using pytest
	cd $(BACKEND_DIR) && env -u FORCE_COLOR uv run nox -s test

test-cov: ## Run tests with coverage report
	cd $(BACKEND_DIR) && uv run pytest --cov=src --cov-report=html --cov-report=term

docker-dev-up: ## Start development Docker services
	docker compose --env-file $(DEVELOPMENT_ENV) -f $(INFRA_DEV_COMPOSE) up -d

docker-dev-down: ## Stop development Docker services
	docker compose --env-file $(DEVELOPMENT_ENV) -f $(INFRA_DEV_COMPOSE) down -v

docker-dev-logs: ## View development Docker services logs
	docker compose --env-file $(DEVELOPMENT_ENV) -f $(INFRA_DEV_COMPOSE) logs -f

docker-dev-restart: ## Restart development Docker services
	docker compose --env-file $(DEVELOPMENT_ENV) -f $(INFRA_DEV_COMPOSE) restart

docker-dev-pause: ## Pause development Docker services
	docker compose --env-file $(DEVELOPMENT_ENV) -f $(INFRA_DEV_COMPOSE) stop

docker-up: ## Start production Docker services
	docker compose --env-file $(PRODUCTION_ENV) -f $(INFRA_COMPOSE) up -d

docker-down: ## Stop production Docker services
	docker compose --env-file $(PRODUCTION_ENV) -f $(INFRA_COMPOSE) down

docker-logs: ## View production Docker services logs
	docker compose --env-file $(PRODUCTION_ENV) -f $(INFRA_COMPOSE) logs -f

docker-restart: ## Restart production Docker services
	docker compose --env-file $(PRODUCTION_ENV) -f $(INFRA_COMPOSE) restart

docker-rebuild-frontend: ## Rebuild and restart the production frontend service
	docker compose --env-file $(PRODUCTION_ENV) -f $(INFRA_COMPOSE) up -d --build --no-deps nginx

docker-rebuild-backend: ## Rebuild and restart the production backend service
	docker compose --env-file $(PRODUCTION_ENV) -f $(INFRA_COMPOSE) up -d --build --no-deps point-source

migration-create: ## Create a new migration (usage: make migration-create MESSAGE="migration message")
	@if [ -z "$(MESSAGE)" ]; then \
		echo "Error: MESSAGE is required. Usage: make migration-create MESSAGE=\"your message\""; \
		exit 1; \
	fi
	cd $(BACKEND_DIR) && ENVIRONMENT=development uv run alembic revision --autogenerate -m "$(MESSAGE)"

migration-upgrade: ## Upgrade database to the latest migration
	cd $(BACKEND_DIR) && ENVIRONMENT=development uv run alembic upgrade head

migration-downgrade: ## Downgrade database by one revision
	cd $(BACKEND_DIR) && ENVIRONMENT=development uv run alembic downgrade -1

migration-current: ## Show current database revision
	cd $(BACKEND_DIR) && ENVIRONMENT=development uv run alembic current

migration-history: ## Show migration history
	cd $(BACKEND_DIR) && ENVIRONMENT=development uv run alembic history

pre-commit-install: ## Install pre-commit hooks
	cd $(BACKEND_DIR) && uv run pre-commit install

pre-commit-run: ## Run pre-commit hooks on all files
	cd $(BACKEND_DIR) && uv run pre-commit run --all-files

createsuperuser: ## Create a superuser account (interactive)
	cd $(BACKEND_DIR) && uv run python -m src.cli createsuperuser
