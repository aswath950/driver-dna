"""Seed the ``circuits`` table from ``data/circuits.json``.

Idempotent — re-running the command upserts each row by ``name`` so circuit
geometry can be refreshed without losing FK references from ``events``.

CLI usage:

    python -m app.etl seed-circuits
    python -m app.etl seed-circuits --path /custom/path/to/circuits.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Circuit
from app.etl.upserts import upsert_one

logger = logging.getLogger(__name__)


# Repo layout: <repo>/backend/app/etl/seed_circuits.py  →  <repo>/data/circuits.json
_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data" / "circuits.json"


@dataclass
class SeedResult:
    path: str
    circuits_seeded: int

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "circuits_seeded": self.circuits_seeded}


def _load_json(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"circuits json not found at {path}")
    with path.open() as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"expected top-level object in {path}, got {type(data).__name__}")
    return data


def run(*, path: Path | None = None) -> SeedResult:
    """Upsert one ``Circuit`` row per top-level key in the source JSON.

    Each value must be a dict with optional ``x``, ``y``, ``sector_fractions``
    keys; missing keys upsert NULL.
    """
    src = path or _DEFAULT_PATH
    data = _load_json(src)

    engine = create_engine(settings.DATABASE_URL_SYNC, future=True)
    seeded = 0
    with Session(engine) as db:
        try:
            for name, body in data.items():
                values = {
                    "name": name,
                    "x": body.get("x"),
                    "y": body.get("y"),
                    "sector_fractions": body.get("sector_fractions"),
                }
                upsert_one(
                    db,
                    Circuit.__table__,
                    values=values,
                    conflict_cols=["name"],
                    update_cols=["x", "y", "sector_fractions"],
                )
                seeded += 1
            db.commit()
        except Exception:
            db.rollback()
            raise

    logger.info("etl.seed_circuits.done path=%s count=%d", src, seeded)
    return SeedResult(path=str(src), circuits_seeded=seeded)
