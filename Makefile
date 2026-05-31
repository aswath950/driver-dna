PYTHON  := .venv/bin/python
PIP     := .venv/bin/pip
RUFF    := .venv/bin/ruff
MYPY    := .venv/bin/mypy
PYTEST  := .venv/bin/pytest

.PHONY: help install-dev lint type-check test ci \
        db backend-install backend backend-test backend-lint \
        web-install web web-build web-typecheck \
        dev hydrate migrate

help:
	@echo "=== Legacy (src/ + Streamlit) ==="
	@echo "  install-dev      Install src/ dev dependencies into .venv"
	@echo "  lint             Run ruff on src/ and tests/"
	@echo "  type-check       Run mypy on race_engine.py and openf1.py"
	@echo "  test             Run pytest with coverage"
	@echo "  ci               lint + type-check + test"
	@echo ""
	@echo "=== Backend (FastAPI + Postgres) ==="
	@echo "  db               Start Postgres + pgAdmin via docker compose (detached)"
	@echo "  backend-install  pip install -e backend[dev] into backend/.venv"
	@echo "  backend          Run uvicorn dev server on :8000"
	@echo "  backend-test     pytest the backend test suite"
	@echo "  backend-lint     ruff backend"
	@echo "  migrate          alembic upgrade head (Phase 2+)"
	@echo "  hydrate          ETL one race weekend into Postgres (Phase 5+)"
	@echo ""
	@echo "=== Web (Next.js) ==="
	@echo "  web-install      pnpm install in web/"
	@echo "  web              pnpm dev (localhost:3000)"
	@echo "  web-build        pnpm build"
	@echo "  web-typecheck    tsc --noEmit"
	@echo ""
	@echo "=== Full stack ==="
	@echo "  dev              db + pgAdmin + backend + web in one command"

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt

lint:
	$(RUFF) check src/ tests/

type-check:
	$(MYPY) src/race_engine.py src/openf1.py \
		--ignore-missing-imports \
		--no-error-summary

test:
	$(PYTEST) tests/ \
		--cov=src \
		--cov-report=xml \
		--cov-report=term-missing \
		-v

ci: lint type-check test

# ---------------------------------------------------------------------------
# Backend (FastAPI)
# ---------------------------------------------------------------------------

BACKEND_VENV := backend/.venv

db:
	docker compose up -d postgres pgadmin

backend-install:
	cd backend && python -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install -e ".[dev]"

backend:
	cd backend && .venv/bin/uvicorn app.main:app --reload --reload-include="*.env" --port 8000

backend-test:
	cd backend && .venv/bin/pytest -v

backend-lint:
	cd backend && .venv/bin/ruff check app tests

migrate:
	@echo "Waiting for Postgres to be ready..."
	@until docker compose exec postgres pg_isready -U dna -d driver_dna > /dev/null 2>&1; do sleep 1; done
	cd backend && .venv/bin/alembic upgrade head

hydrate:
	@echo "Phase 5+ — usage: make hydrate YEAR=2024 GP='Monaco' SESSION=R"
	cd backend && .venv/bin/python -m app.etl hydrate --year $(YEAR) --gp "$(GP)" --session $(SESSION)

# ---------------------------------------------------------------------------
# Web (Next.js)
# ---------------------------------------------------------------------------

web-install:
	cd web && pnpm install

web:
	cd web && pnpm dev

web-build:
	cd web && pnpm build

web-typecheck:
	cd web && pnpm typecheck

# ---------------------------------------------------------------------------
# Full stack (best-effort; prefer 3 terminals during real dev)
# ---------------------------------------------------------------------------

dev:
	@echo "Starting Postgres + pgAdmin..."
	docker compose up -d postgres pgadmin
	@echo "Waiting for Postgres to be healthy..."
	@until docker compose exec postgres pg_isready -U dna -d driver_dna > /dev/null 2>&1; do sleep 1; done
	@echo "Starting FastAPI backend (background)..."
	cd backend && .venv/bin/uvicorn app.main:app --reload --port 8000 &
	@echo "Starting Next.js web (foreground — Ctrl-C stops all)..."
	cd web && pnpm dev
