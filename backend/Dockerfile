# Used by docker-compose (local dev) and Railway (production).
# Build context MUST be the repo root, so we can COPY both ./backend and ./src.
#
#   docker build -f backend/Dockerfile -t driver-dna-backend .
#
# Railway: set "Root Directory" to the repo root and "Dockerfile path" to
# backend/Dockerfile so this layout works.

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# System deps for asyncpg/psycopg builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

# 1) Install Python deps from backend/pyproject.toml (cacheable layer).
COPY backend/pyproject.toml /app/backend/pyproject.toml
RUN pip install --upgrade pip && pip install -e /app/backend

# 2) Copy shared src/ and backend code.
COPY src /app/src
COPY backend /app/backend

EXPOSE 8000

# Railway sets $PORT; default to 8000 for docker-compose.
ENV PORT=8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://localhost:${PORT}/healthz || exit 1

# `release` runs alembic upgrade head, then start uvicorn.
CMD ["sh", "-c", "cd /app/backend && alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 2"]
