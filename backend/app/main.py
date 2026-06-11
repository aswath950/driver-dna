from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Note: sys.path insertion for the shared `src/` library happens in
# app/__init__.py so it fires on any `app.*` entry point (uvicorn, alembic,
# pytest, python -m app.etl).
from app.api.v1 import router as api_v1_router  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.core.errors import register_exception_handlers  # noqa: E402
from app.core.logging import configure_logging, get_logger  # noqa: E402
from app.core.middleware import register_middleware  # noqa: E402
from app.graphql.schema import build_graphql_router  # noqa: E402

logger = get_logger(__name__)


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(
        title=settings.API_TITLE,
        version=settings.API_VERSION,
        # FastAPI 0.115+ emits OpenAPI 3.1.0 by default.
    )

    # CORS first so preflight requests don't trip exception handlers.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "API-Version"],
    )

    register_middleware(app)
    register_exception_handlers(app)

    app.include_router(api_v1_router)
    app.include_router(build_graphql_router(), prefix="/graphql")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        # DB check is added in later phases; for now we just report the env.
        return {"status": "ok", "env": settings.ENV, "version": settings.API_VERSION}

    @app.get("/")
    async def root() -> dict[str, str]:
        return {
            "name": settings.API_TITLE,
            "docs": "/docs",
            "openapi": "/openapi.json",
            "health": "/healthz",
            "api_v1": "/api/v1/_ping",
        }

    logger.info("app.startup", env=settings.ENV, version=settings.API_VERSION)
    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=settings.ENV == "local",
    )
