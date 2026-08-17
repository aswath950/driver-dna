PIP     := .venv/bin/pip
RUFF    := .venv/bin/ruff
MYPY    := .venv/bin/mypy
PYTEST  := .venv/bin/pytest

.PHONY: help \
        install-dev lint type-check test ci \
        backend-install backend backend-test backend-lint \
        web-install web web-build web-typecheck \
        db migrate seed-circuits seed-corners refresh-stats fetch-telemetry hydrate \
        monitoring monitoring-down dev

help:
	@echo "DriverDNA runs as one of two apps — pick an option:"
	@echo ""
	@echo "=== OPTION 1 · Legacy Streamlit (standalone dashboard) ==="
	@echo "  Run manually:  streamlit run streamlit_app.py   (-> :8501)"
	@echo "  Deploy:        Streamlit Cloud, or the root Dockerfile"
	@echo "  install-dev      Install src/ (Streamlit) deps into .venv"
	@echo "  lint             Run ruff on src/ and tests/"
	@echo "  type-check       Run mypy on race_engine.py and openf1.py"
	@echo "  test             Run pytest with coverage"
	@echo "  ci               lint + type-check + test"
	@echo ""
	@echo "=== OPTION 2 · Full-stack (FastAPI + Next.js + Postgres) ==="
	@echo "  Deploy:        Vercel (web) + Railway (backend), or docker compose"
	@echo "  - Backend (FastAPI):"
	@echo "  backend-install  pip install -e backend[dev] into backend/.venv"
	@echo "  backend          Run uvicorn dev server on :8000"
	@echo "  backend-test     pytest the backend test suite"
	@echo "  backend-lint     ruff backend"
	@echo "  - Web (Next.js):"
	@echo "  web-install      pnpm install in web/"
	@echo "  web              pnpm dev (localhost:3000)"
	@echo "  web-build        pnpm build"
	@echo "  web-typecheck    tsc --noEmit"
	@echo "  - Database & ETL:"
	@echo "  db               Start Postgres + pgAdmin via docker compose (detached)"
	@echo "  migrate          alembic upgrade head + seed circuit corners"
	@echo "  seed-circuits    Upsert circuit geometry (CIRCUITS_PATH= optional)"
	@echo "  seed-corners     Seed FastF1 corner data only (YEAR= optional)"
	@echo "  refresh-stats    Recompute driver_stats (SEASON=2024 or ALL_SEASONS=1)"
	@echo "  fetch-telemetry  Download + cache telemetry (SESSION_ID= required)"
	@echo "  hydrate          ETL one race weekend into Postgres (YEAR= GP= SESSION=)"
	@echo "  - Metrics:"
	@echo "  monitoring       Prometheus + postgres_exporter + Grafana (-> :3001)"
	@echo "  monitoring-down  Stop the metrics stack"
	@echo "  - Run it all:"
	@echo "  dev              db + pgAdmin + backend + web in one command"

# ---------------------------------------------------------------------------
# OPTION 1 — Legacy Streamlit (standalone)
# ---------------------------------------------------------------------------

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
# OPTION 2 — Full-stack · Backend (FastAPI + Postgres)
# ---------------------------------------------------------------------------

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

# OPTION 2 — Full-stack · Database & ETL
migrate:
	@echo "Waiting for Postgres to be ready..."
	@until docker compose exec postgres pg_isready -U dna -d driver_dna > /dev/null 2>&1; do sleep 1; done
	cd backend && .venv/bin/alembic upgrade head
	@echo "Seeding circuit corner data from FastF1$(if $(YEAR), (preferred year=$(YEAR)),)..."
	cd backend && .venv/bin/python -m app.etl seed-circuit-corners $(if $(YEAR),--year $(YEAR),)

seed-corners:
	@echo "Seeding circuit corner data from FastF1$(if $(YEAR), (preferred year=$(YEAR)),)..."
	cd backend && .venv/bin/python -m app.etl seed-circuit-corners $(if $(YEAR),--year $(YEAR),)

hydrate:
	@echo "Usage: make hydrate YEAR=2024 GP='Monaco' SESSION=R"
	cd backend && .venv/bin/python -m app.etl hydrate --year $(YEAR) --gp "$(GP)" --session $(SESSION)

seed-circuits:
	@echo "Seeding circuit geometry from data/circuits.json..."
	cd backend && .venv/bin/python -m app.etl seed-circuits $(if $(CIRCUITS_PATH),--path "$(CIRCUITS_PATH)",)

refresh-stats:
	@echo "Refreshing driver stats..."
	cd backend && .venv/bin/python -m app.etl refresh-stats $(if $(ALL_SEASONS),--all-seasons,--season $(SEASON))

fetch-telemetry:
	@echo "Fetching telemetry for session $(SESSION_ID)..."
	cd backend && .venv/bin/python -m app.etl fetch-telemetry --session-id $(SESSION_ID)

# OPTION 2 — Full-stack · Metrics (Prometheus + postgres_exporter + Grafana)
monitoring:
	docker compose --profile monitoring up -d postgres postgres_exporter prometheus grafana
	@echo "Grafana    -> http://localhost:3001  (admin / admin)"
	@echo "Prometheus -> http://localhost:9090"

monitoring-down:
	docker compose --profile monitoring stop postgres_exporter prometheus grafana

# ---------------------------------------------------------------------------
# OPTION 2 — Full-stack · Web (Next.js)
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
# OPTION 2 — Full-stack · Run it all (best-effort; prefer 3 terminals)
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
