"""Tests for the ``seed-circuits`` ETL command.

Writes a tiny synthetic circuits.json to a tmp_path and verifies:
- one Circuit row per top-level key is inserted with x/y/sector_fractions
- running the seeder twice is a no-op (row count stable, idempotent)
- editing the source JSON and re-running refreshes the existing row in place
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import sqlalchemy as sa

from app.etl import seed_circuits


_SAMPLE = {
    "Test Grand Prix": {
        "x": [0.0, 1.0, 2.0],
        "y": [10.0, 11.0, 12.0],
        "sector_fractions": [0.33, 0.66],
    },
    "Another Grand Prix": {
        "x": [5.0, 6.0],
        "y": [50.0, 60.0],
        "sector_fractions": [0.4, 0.8],
    },
}


def _write_json(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "circuits.json"
    p.write_text(json.dumps(payload))
    return p


def _read_circuits(engine: sa.Engine) -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(
            sa.text("SELECT name, x, y, sector_fractions FROM circuits ORDER BY name")
        ).all()
    return [dict(r._mapping) for r in rows]


def test_seed_circuits_inserts_rows(clean_db: sa.Engine, tmp_path: Path) -> None:
    src = _write_json(tmp_path, _SAMPLE)
    result = seed_circuits.run(path=src)
    assert result.circuits_seeded == 2

    rows = _read_circuits(clean_db)
    assert [r["name"] for r in rows] == ["Another Grand Prix", "Test Grand Prix"]
    test_row = next(r for r in rows if r["name"] == "Test Grand Prix")
    assert test_row["x"] == [0.0, 1.0, 2.0]
    assert test_row["y"] == [10.0, 11.0, 12.0]
    assert test_row["sector_fractions"] == [0.33, 0.66]


def test_seed_circuits_is_idempotent(clean_db: sa.Engine, tmp_path: Path) -> None:
    src = _write_json(tmp_path, _SAMPLE)
    seed_circuits.run(path=src)
    first = _read_circuits(clean_db)
    seed_circuits.run(path=src)
    second = _read_circuits(clean_db)
    assert first == second


def test_seed_circuits_refreshes_changed_fields(
    clean_db: sa.Engine, tmp_path: Path
) -> None:
    src = _write_json(tmp_path, _SAMPLE)
    seed_circuits.run(path=src)

    updated = {
        "Test Grand Prix": {
            "x": [9.0, 9.5],
            "y": [99.0, 99.5],
            "sector_fractions": [0.5, 0.75],
        }
    }
    src.write_text(json.dumps(updated))
    seed_circuits.run(path=src)

    rows = _read_circuits(clean_db)
    test_row = next(r for r in rows if r["name"] == "Test Grand Prix")
    assert test_row["x"] == [9.0, 9.5]
    assert test_row["sector_fractions"] == [0.5, 0.75]


def test_seed_circuits_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        seed_circuits.run(path=tmp_path / "does-not-exist.json")
