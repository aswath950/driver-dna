"""
test_openf1.py — Unit tests for openf1.py

All HTTP calls are intercepted by the ``responses`` library, so
these tests run fully offline with no real network access.
"""

import requests
import unittest.mock
from unittest.mock import patch

import pandas as pd
import pytest
import responses as responses_lib

from openf1 import BASE_URL, OpenF1Client, validate_dataframe


# ===========================================================================
# validate_dataframe (standalone helper)
# ===========================================================================

class TestValidateDataframe:
    def test_missing_column_filled_with_nan(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = validate_dataframe(df, ["a", "b", "c"])
        assert "c" in result.columns
        assert result["c"].isna().all()

    def test_empty_df_returned_unchanged(self):
        df = pd.DataFrame()
        result = validate_dataframe(df, ["a", "b"])
        assert result.empty

    def test_present_columns_are_not_modified(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        result = validate_dataframe(df, ["a", "b"])
        assert list(result["a"]) == [1]

    def test_multiple_missing_columns_all_filled(self):
        df = pd.DataFrame({"x": [10]})
        result = validate_dataframe(df, ["x", "y", "z"])
        assert "y" in result.columns
        assert "z" in result.columns


# ===========================================================================
# OpenF1Client initialisation
# ===========================================================================

class TestOpenF1ClientInit:
    def test_historical_mode_stores_mode(self):
        client = OpenF1Client(mode="historical")
        assert client.mode == "historical"

    def test_live_mode_stores_mode(self):
        client = OpenF1Client(mode="live")
        assert client.mode == "live"

    def test_invalid_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="mode must be"):
            OpenF1Client(mode="replay")

    def test_live_watermarks_initialised_as_none(self):
        client = OpenF1Client(mode="live")
        assert client._last_lap_ts is None
        assert client._last_stint_ts is None
        assert client._last_position_ts is None


# ===========================================================================
# _require_live guard
# ===========================================================================

class TestRequireLive:
    def test_raises_in_historical_mode(self):
        client = OpenF1Client(mode="historical")
        with pytest.raises(RuntimeError, match="Live methods"):
            client._require_live()

    def test_passes_silently_in_live_mode(self):
        client = OpenF1Client(mode="live")
        client._require_live()  # must not raise

    def test_get_live_laps_raises_in_historical_mode(self):
        client = OpenF1Client(mode="historical")
        with pytest.raises(RuntimeError):
            client.get_live_laps()

    def test_get_live_stints_raises_in_historical_mode(self):
        client = OpenF1Client(mode="historical")
        with pytest.raises(RuntimeError):
            client.get_live_stints()


# ===========================================================================
# get_laps (historical mode, mocked HTTP)
# ===========================================================================

class TestGetLaps:
    @responses_lib.activate
    def test_success_returns_dataframe(self):
        responses_lib.add(
            responses_lib.GET,
            f"{BASE_URL}/laps",
            json=[{
                "driver_number": 1, "lap_number": 1, "lap_duration": 90.123,
                "is_pit_out_lap": False, "st_speed": 310, "session_key": 9999,
                "date_start": "2026-04-06T05:01:30",
            }],
            status=200,
        )
        client = OpenF1Client(mode="historical")
        result = client.get_laps(session_key=9999)
        assert not result.empty
        assert result.iloc[0]["driver_number"] == 1
        assert abs(result.iloc[0]["lap_duration"] - 90.123) < 1e-6

    @responses_lib.activate
    def test_missing_column_filled_with_nan(self):
        # API response omits 'st_speed'
        responses_lib.add(
            responses_lib.GET,
            f"{BASE_URL}/laps",
            json=[{
                "driver_number": 1, "lap_number": 1, "lap_duration": 90.0,
                "is_pit_out_lap": False, "session_key": 9999,
                "date_start": "2026-04-06T05:01:30",
            }],
            status=200,
        )
        client = OpenF1Client(mode="historical")
        result = client.get_laps(session_key=9999)
        assert "st_speed" in result.columns
        assert pd.isna(result.iloc[0]["st_speed"])

    @responses_lib.activate
    @patch("openf1.time.sleep")
    def test_all_retries_fail_returns_empty(self, mock_sleep):
        # Register 3 connection errors (one per retry attempt)
        for _ in range(3):
            responses_lib.add(
                responses_lib.GET,
                f"{BASE_URL}/laps",
                body=requests.ConnectionError("Connection refused"),
            )
        client = OpenF1Client(mode="historical", timeout=1)
        result = client.get_laps(session_key=9999)
        assert result.empty

    @responses_lib.activate
    @patch("openf1.time.sleep")
    def test_retry_succeeds_on_third_attempt(self, mock_sleep):
        # First two attempts fail, third succeeds
        for _ in range(2):
            responses_lib.add(
                responses_lib.GET,
                f"{BASE_URL}/laps",
                body=requests.ConnectionError("Temporary failure"),
            )
        responses_lib.add(
            responses_lib.GET,
            f"{BASE_URL}/laps",
            json=[{
                "driver_number": 2, "lap_number": 1, "lap_duration": 91.0,
                "is_pit_out_lap": False, "st_speed": 305, "session_key": 9999,
                "date_start": "2026-04-06T05:01:30",
            }],
            status=200,
        )
        client = OpenF1Client(mode="historical", timeout=1)
        result = client.get_laps(session_key=9999)
        assert not result.empty
        assert result.iloc[0]["driver_number"] == 2
        # sleep was called between attempt 1→2 and 2→3
        assert mock_sleep.call_count == 2


# ===========================================================================
# get_sessions (historical mode, mocked HTTP)
# ===========================================================================

class TestGetSessions:
    @responses_lib.activate
    def test_empty_meeting_returns_empty_dataframe(self):
        responses_lib.add(
            responses_lib.GET,
            f"{BASE_URL}/meetings",
            json=[],
            status=200,
        )
        client = OpenF1Client(mode="historical")
        result = client.get_sessions(year=2025, grand_prix="Nonexistent GP")
        assert result.empty

    @responses_lib.activate
    def test_success_returns_session_data(self):
        responses_lib.add(
            responses_lib.GET,
            f"{BASE_URL}/meetings",
            json=[{"meeting_key": 123, "meeting_name": "Japanese Grand Prix"}],
            status=200,
        )
        responses_lib.add(
            responses_lib.GET,
            f"{BASE_URL}/sessions",
            json=[{
                "session_key": 9001, "session_name": "Race",
                "session_type": "Race",
                "date_start": "2026-04-06T07:00:00",
                "date_end": "2026-04-06T09:00:00",
            }],
            status=200,
        )
        client = OpenF1Client(mode="historical")
        result = client.get_sessions(year=2026, grand_prix="Japanese Grand Prix")
        assert not result.empty
        assert "session_key" in result.columns
        assert result.iloc[0]["session_key"] == 9001
