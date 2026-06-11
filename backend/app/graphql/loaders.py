"""Per-request DataLoaders so nested fields don't trigger N+1 queries.

A new loader bundle is created for each GraphQL request via
``context_getter`` in :mod:`app.graphql.schema`. Resolvers reach for
``info.context["loaders"].drivers.load(id)`` etc.; the loader batches
all ``.load()`` calls within a single tick into one SQL fetch.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from strawberry.dataloader import DataLoader

from app.db.models import Circuit, Driver, Team


async def _batch_drivers(
    db: AsyncSession, ids: list[int]
) -> list[Driver | None]:
    rows = (
        await db.execute(select(Driver).where(Driver.id.in_(ids)))
    ).scalars().all()
    by_id = {r.id: r for r in rows}
    return [by_id.get(i) for i in ids]


async def _batch_teams(
    db: AsyncSession, ids: list[int]
) -> list[Team | None]:
    rows = (
        await db.execute(select(Team).where(Team.id.in_(ids)))
    ).scalars().all()
    by_id = {r.id: r for r in rows}
    return [by_id.get(i) for i in ids]


async def _batch_circuits(
    db: AsyncSession, ids: list[int]
) -> list[Circuit | None]:
    rows = (
        await db.execute(select(Circuit).where(Circuit.id.in_(ids)))
    ).scalars().all()
    by_id = {r.id: r for r in rows}
    return [by_id.get(i) for i in ids]


@dataclass
class Loaders:
    drivers: DataLoader[int, Driver | None]
    teams: DataLoader[int, Team | None]
    circuits: DataLoader[int, Circuit | None]


def build_loaders(db: AsyncSession) -> Loaders:
    return Loaders(
        drivers=DataLoader(load_fn=lambda ids: _batch_drivers(db, ids)),
        teams=DataLoader(load_fn=lambda ids: _batch_teams(db, ids)),
        circuits=DataLoader(load_fn=lambda ids: _batch_circuits(db, ids)),
    )
