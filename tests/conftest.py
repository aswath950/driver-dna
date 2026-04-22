"""
conftest.py — Shared pytest fixtures for driver-dna tests.

All fixtures use small synthetic DataFrames so tests run without
the real dataset.parquet or any network access.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make src/ importable without a package install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Lap data — 3 drivers × 5 laps, clean (no pit-out laps)
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_laps() -> pd.DataFrame:
    """Synthetic laps DataFrame for 3 drivers, 5 laps each.

    Driver 1 is fastest (~90 s/lap), driver 3 slowest (~92 s/lap).
    Each row includes a ``date`` column so position merging works.
    """
    base_time = pd.Timestamp("2026-04-06 05:00:00", tz="UTC")
    base_pace = {1: 90.0, 2: 91.0, 3: 92.0}
    rows = []
    for driver_number, pace in base_pace.items():
        cumulative = 0.0
        for lap in range(1, 6):
            lap_time = pace + lap * 0.1  # slight degradation per lap
            cumulative += lap_time
            rows.append({
                "driver_number": driver_number,
                "lap_number": lap,
                "lap_duration": lap_time,
                "is_pit_out_lap": False,
                "date": base_time + pd.Timedelta(seconds=cumulative),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stint data — driver 2 pits after lap 3
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_stints() -> pd.DataFrame:
    """Stints: driver 1 (SOFT, 1-5), driver 2 pits lap 3→4 (MEDIUM→HARD),
    driver 3 (HARD, 1-5)."""
    return pd.DataFrame([
        {"driver_number": 1, "stint_number": 1, "compound": "SOFT",
         "tyre_age_at_start": 0, "lap_start": 1, "lap_end": 5},
        {"driver_number": 2, "stint_number": 1, "compound": "MEDIUM",
         "tyre_age_at_start": 0, "lap_start": 1, "lap_end": 3},
        {"driver_number": 2, "stint_number": 2, "compound": "HARD",
         "tyre_age_at_start": 0, "lap_start": 4, "lap_end": 5},
        {"driver_number": 3, "stint_number": 1, "compound": "HARD",
         "tyre_age_at_start": 0, "lap_start": 1, "lap_end": 5},
    ])


# ---------------------------------------------------------------------------
# Position data — positions match driver numbers (1=P1, 2=P2, 3=P3)
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_position(minimal_laps) -> pd.DataFrame:
    """One position row per lap row; position = driver_number for simplicity."""
    rows = []
    for _, lap_row in minimal_laps.iterrows():
        rows.append({
            "driver_number": lap_row["driver_number"],
            "position": int(lap_row["driver_number"]),
            "date": lap_row["date"],
            "session_key": 9999,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fully assembled RaceAnalyser
# ---------------------------------------------------------------------------

@pytest.fixture
def analyser(minimal_laps, minimal_stints, minimal_position):
    """RaceAnalyser backed by the minimal synthetic DataFrames."""
    from race_engine import RaceAnalyser
    return RaceAnalyser(minimal_laps, minimal_stints, minimal_position)


# ---------------------------------------------------------------------------
# Empty DataFrames for edge-case tests
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_laps() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "driver_number", "lap_number", "lap_duration", "is_pit_out_lap",
    ])


@pytest.fixture
def empty_stints() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "driver_number", "stint_number", "compound",
        "tyre_age_at_start", "lap_start", "lap_end",
    ])


@pytest.fixture
def empty_position() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "driver_number", "position", "date", "session_key",
    ])
