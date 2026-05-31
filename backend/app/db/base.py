from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Shared declarative base. All ORM models inherit from this so Alembic
    can pick them up via ``Base.metadata``.
    """

    pass
