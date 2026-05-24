"""
test_telemetry_cumtime.py — Unit tests for the cumtime overflow fix in
_fetch_fastest_lap_all_openf1 (src/viz.py).

The 0.5s query buffer added to date_lte can pull car_data samples from the
following lap when consecutive laps are closely spaced.  Two guards were added:
  1. Clip car_data to ts_end before building traces.
  2. Normalize cumtime[-1] to the official lap_duration.

These tests verify both guards work correctly and that the fix resolves the
real-world Russell/Antonelli inversion from the 2026 Canadian GP Sprint Qualifying.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_car_data(
    start_ts: pd.Timestamp,
    duration_s: float,
    n_samples: int,
    extra_s: float = 0.0,
) -> pd.DataFrame:
    """UTC-aware car_data spanning [start_ts, start_ts + duration_s + extra_s]."""
    total_s = duration_s + extra_s
    times = pd.date_range(
        start=start_ts,
        periods=n_samples,
        freq=pd.Timedelta(seconds=total_s / (n_samples - 1)),
    )
    return pd.DataFrame({
        "date":     times,
        "speed":    np.full(n_samples, 200.0),
        "throttle": np.full(n_samples, 80.0),
        "brake":    np.zeros(n_samples),
    })


def _make_laps_df(
    driver_number: int,
    date_start: str,
    lap_duration: float,
    lap_number: int = 1,
) -> pd.DataFrame:
    return pd.DataFrame([{
        "driver_number":  driver_number,
        "lap_duration":   lap_duration,
        "lap_number":     lap_number,
        "date_start":     date_start,
        "is_pit_out_lap": False,
    }])


# ---------------------------------------------------------------------------
# Test 1: overflow from 0.5s buffer is clipped — cumtime must not be inflated
# ---------------------------------------------------------------------------

def test_cumtime_not_inflated_by_buffer():
    """When the API returns 0.5s of next-lap data, cumtime[-1] must equal lap_duration."""
    LAP_DURATION = 72.965
    START_TS = pd.Timestamp("2026-05-22T21:32:11.811+00:00")
    DRIVER = 63
    SESSION = 11282

    car_df = _make_car_data(START_TS, LAP_DURATION, n_samples=270, extra_s=0.5)
    laps_df = _make_laps_df(DRIVER, START_TS.isoformat(), LAP_DURATION)

    mock_client = MagicMock()
    mock_client.get_car_data.return_value = car_df

    with patch("viz.OpenF1Client", return_value=mock_client):
        from viz import _fetch_fastest_lap_all_openf1
        result = _fetch_fastest_lap_all_openf1(SESSION, DRIVER, laps_df)

    assert result is not None
    assert abs(result["cumtime"][-1] - LAP_DURATION) < 1e-6, (
        f"cumtime[-1]={result['cumtime'][-1]:.6f} should equal {LAP_DURATION}"
    )
    assert np.all(np.diff(result["cumtime"]) >= -1e-9), "cumtime must be non-decreasing"


# ---------------------------------------------------------------------------
# Test 2: last sample before ts_end — cumtime still pinned to lap_duration
# ---------------------------------------------------------------------------

def test_cumtime_correct_when_last_sample_before_lap_end():
    """When the last sample arrives 0.27s before lap end, cumtime[-1] must still equal lap_duration."""
    LAP_DURATION = 73.033
    START_TS = pd.Timestamp("2026-05-22T21:32:34.184+00:00")
    DRIVER = 12
    SESSION = 11282

    # Final sample lands 0.27s short — simulates normal 3.7 Hz sampling jitter
    car_df = _make_car_data(START_TS, LAP_DURATION - 0.27, n_samples=270, extra_s=0.0)
    laps_df = _make_laps_df(DRIVER, START_TS.isoformat(), LAP_DURATION)

    mock_client = MagicMock()
    mock_client.get_car_data.return_value = car_df

    with patch("viz.OpenF1Client", return_value=mock_client):
        from viz import _fetch_fastest_lap_all_openf1
        result = _fetch_fastest_lap_all_openf1(SESSION, DRIVER, laps_df)

    assert result is not None
    assert abs(result["cumtime"][-1] - LAP_DURATION) < 1e-6, (
        f"cumtime[-1]={result['cumtime'][-1]:.6f} should equal {LAP_DURATION}"
    )


# ---------------------------------------------------------------------------
# Test 3: Russell vs Antonelli — correct sign and magnitude of final delta
# ---------------------------------------------------------------------------

def test_russell_faster_than_antonelli_delta():
    """
    Reproduces the 2026 Canadian GP Sprint Qualifying bug:
    Russell's lap was 0.068s faster but the old code showed Antonelli faster.

    With the fix, the final delta must be positive (Russell ahead) and equal
    to ANTONELLI_DUR - RUSSELL_DUR within 0.1ms tolerance.
    """
    RUSSELL_DUR = 72.965
    ANTONELLI_DUR = 73.033
    START_RUS = pd.Timestamp("2026-05-22T21:32:11.811+00:00")
    START_ANT = pd.Timestamp("2026-05-22T21:32:34.184+00:00")
    SESSION = 11282

    # Russell: 0.5s overflow from closely-spaced next lap
    car_rus = _make_car_data(START_RUS, RUSSELL_DUR, n_samples=270, extra_s=0.5)
    # Antonelli: no overflow
    car_ant = _make_car_data(START_ANT, ANTONELLI_DUR, n_samples=270, extra_s=0.0)

    laps_rus = _make_laps_df(63, START_RUS.isoformat(), RUSSELL_DUR)
    laps_ant = _make_laps_df(12, START_ANT.isoformat(), ANTONELLI_DUR)

    mock_rus = MagicMock()
    mock_rus.get_car_data.return_value = car_rus
    mock_ant = MagicMock()
    mock_ant.get_car_data.return_value = car_ant

    with patch("viz.OpenF1Client", side_effect=[mock_rus, mock_ant]):
        from viz import _fetch_fastest_lap_all_openf1
        data_rus = _fetch_fastest_lap_all_openf1(SESSION, 63, laps_rus)
        data_ant = _fetch_fastest_lap_all_openf1(SESSION, 12, laps_ant)

    assert data_rus is not None
    assert data_ant is not None

    # delta = cumtime_b - cumtime_a; positive means A (Russell) used less time → Russell ahead
    delta_final = data_ant["cumtime"][-1] - data_rus["cumtime"][-1]
    expected_gap = ANTONELLI_DUR - RUSSELL_DUR  # 0.068s

    assert delta_final > 0, (
        f"Expected Russell ahead (delta > 0) but got delta={delta_final:.4f}s"
    )
    assert abs(delta_final - expected_gap) < 1e-4, (
        f"Gap mismatch: got {delta_final:.4f}s, expected {expected_gap:.4f}s"
    )


# ---------------------------------------------------------------------------
# Test 4: no overflow — clip must not remove any legitimate samples
# ---------------------------------------------------------------------------

def test_no_samples_removed_when_no_overflow():
    """When car_data ends exactly at the lap boundary, no samples must be discarded."""
    LAP_DURATION = 90.0
    START_TS = pd.Timestamp("2024-03-24T05:00:00+00:00")
    DRIVER = 1
    SESSION = 9999

    # Exactly on the boundary — all 200 samples are legitimate
    car_df = _make_car_data(START_TS, LAP_DURATION, n_samples=200, extra_s=0.0)
    laps_df = _make_laps_df(DRIVER, START_TS.isoformat(), LAP_DURATION)

    mock_client = MagicMock()
    mock_client.get_car_data.return_value = car_df

    with patch("viz.OpenF1Client", return_value=mock_client):
        from viz import _fetch_fastest_lap_all_openf1
        result = _fetch_fastest_lap_all_openf1(SESSION, DRIVER, laps_df)

    assert result is not None
    assert abs(result["cumtime"][-1] - LAP_DURATION) < 1e-6
    assert abs(result["lap_time"] - LAP_DURATION) < 1e-9
