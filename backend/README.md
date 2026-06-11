# Driver DNA — Backend

FastAPI + Strawberry GraphQL + SQLAlchemy 2.0 service. Sibling to the existing
`src/` Python lib (which it imports directly) and the legacy Streamlit app at
the repo root.

## Layout

```
backend/
├── app/
│   ├── main.py             # FastAPI factory
│   ├── core/               # config, logging, errors, pagination, middleware
│   ├── api/v1/             # REST routers + schemas
│   ├── graphql/            # Strawberry schema, resolvers, dataloaders
│   ├── db/                 # SQLAlchemy models, session, repositories
│   ├── etl/                # FastF1/OpenF1 → Postgres hydration jobs
│   └── llm/                # async wrappers over ../src/llm_layer
├── alembic/                # migrations (added in Phase 2)
├── tests/
├── pyproject.toml
└── .env.example
```

## Shared `src/` import

`backend/app/main.py` prepends the repo root to `sys.path` at startup so any
module can `from src.race_engine import RaceAnalyser`. This avoids packaging
`src/` and keeps Streamlit + MCP unaffected.

For pytest / standalone scripts, run from the `backend/` directory with an
activated venv — the same `sys.path` insertion fires when `app` is imported.

## First-time setup

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env             # edit OPENAI_API_KEY etc.

# Postgres (from repo root)
cd .. && docker compose up -d postgres

# Run dev server
cd backend && uvicorn app.main:app --reload --port 8000
curl -s localhost:8000/healthz | jq .
```

## Conventions

- Async everywhere: SQLAlchemy async session, async resolvers, async test fixtures.
- Pydantic v2 models in `app/api/v1/schemas/`; never leak ORM objects across the
  router boundary.
- One repository per aggregate in `app/db/repositories/`; both REST routers and
  GraphQL resolvers call them so SQL lives in exactly one place.
- All errors flow through the RFC 7807 envelope defined in
  `app/core/errors.py` (added in Phase 4).
- Logs are JSON via `structlog`, bound with `request_id` per request.

## Tests

```bash
pytest -v                          # all tests
pytest tests/api -v                # router tests only
pytest --cov=app --cov-report=term # with coverage
```

## Migrations (added in Phase 2)

```bash
alembic upgrade head
alembic revision -m "describe change" --autogenerate
```
