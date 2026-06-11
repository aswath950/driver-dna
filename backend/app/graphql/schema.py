"""Strawberry schema + FastAPI router.

Per-request context contains:
  - ``db``      — the same async session FastAPI dependencies use.
  - ``loaders`` — a fresh DataLoader bundle for this request.

GraphiQL is enabled only when ``settings.ENV == "local"``.
"""

from __future__ import annotations

import strawberry
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from strawberry.fastapi import GraphQLRouter

from app.core.config import settings
from app.core.deps import DB
from app.graphql.loaders import build_loaders
from app.graphql.resolvers import Query

schema = strawberry.Schema(query=Query)


async def get_context(db: AsyncSession = Depends(lambda: None)) -> dict:  # type: ignore[assignment]
    # Bound at request time by FastAPI's dependency-injection — the lambda
    # here is just a placeholder; the real wiring is in ``build_graphql_router``.
    return {"db": db, "loaders": build_loaders(db)}


def build_graphql_router() -> GraphQLRouter:
    """Create the GraphQL router with DB + loaders injected per request."""

    async def _context(db: DB) -> dict:
        return {"db": db, "loaders": build_loaders(db)}

    return GraphQLRouter(
        schema,
        context_getter=_context,
        graphql_ide="graphiql" if settings.ENV == "local" else None,
    )
