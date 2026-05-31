"""Postgres upsert helpers for the ETL.

Every ETL step uses ``INSERT ... ON CONFLICT DO UPDATE`` keyed on a natural
unique constraint. This keeps the entire pipeline idempotent — re-running
the hydrate job for the same race is a true no-op (zero inserts, zero
mutating updates).

These helpers are intentionally small and dialect-coupled (postgres only).
The ETL is the only consumer.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from sqlalchemy import Table
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session


def upsert_one(
    db: Session,
    table: Table,
    *,
    values: dict[str, Any],
    conflict_cols: Sequence[str],
    update_cols: Sequence[str] | None = None,
    returning_cols: Sequence[str] = ("id",),
) -> dict[str, Any]:
    """Insert one row, or update on conflict, and return the resulting row.

    Parameters
    ----------
    table
        SQLAlchemy ``Table`` (use ``Model.__table__``).
    values
        Column → value mapping. Columns not present are left to defaults.
    conflict_cols
        Columns that form the unique constraint to detect conflicts on.
    update_cols
        Columns to update on conflict. Defaults to every non-conflict column
        in ``values`` (so the row is refreshed with the latest data).
    returning_cols
        Columns to RETURN — typically the synthesized PK.
    """
    if update_cols is None:
        update_cols = [c for c in values if c not in conflict_cols]

    stmt = pg_insert(table).values(**values)
    if update_cols:
        excluded = {c: stmt.excluded[c] for c in update_cols}
        stmt = stmt.on_conflict_do_update(index_elements=list(conflict_cols), set_=excluded)
    else:
        # No mutable columns → just no-op on conflict.
        stmt = stmt.on_conflict_do_nothing(index_elements=list(conflict_cols))

    stmt = stmt.returning(*[table.c[col] for col in returning_cols])
    row = db.execute(stmt).first()
    if row is None:
        # on_conflict_do_nothing returns nothing when conflict hit — fetch.
        from sqlalchemy import select

        where = [table.c[col] == values[col] for col in conflict_cols]
        sel = select(*[table.c[col] for col in returning_cols]).where(*where)
        row = db.execute(sel).first()
    assert row is not None, "upsert returned nothing and lookup found nothing"
    return dict(row._mapping)


def upsert_many(
    db: Session,
    table: Table,
    *,
    rows: Iterable[dict[str, Any]],
    conflict_cols: Sequence[str],
    update_cols: Sequence[str] | None = None,
) -> int:
    """Bulk insert/upsert. Returns the number of rows fed in.

    Use this for the high-volume tables (``lap_times``, ``session_drivers``,
    ``race_results``). Does NOT return row IDs — when callers need them they
    should query afterwards by the natural key.
    """
    payload = list(rows)
    if not payload:
        return 0

    # Derive update_cols from the FIRST row's keys (all rows must be uniform).
    first = payload[0]
    if update_cols is None:
        update_cols = [c for c in first if c not in conflict_cols]

    stmt = pg_insert(table).values(payload)
    if update_cols:
        excluded = {c: stmt.excluded[c] for c in update_cols}
        stmt = stmt.on_conflict_do_update(index_elements=list(conflict_cols), set_=excluded)
    else:
        stmt = stmt.on_conflict_do_nothing(index_elements=list(conflict_cols))

    db.execute(stmt)
    return len(payload)
