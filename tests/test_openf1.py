"""
test_openf1.py — Unit tests for openf1.py

All HTTP calls are intercepted by the ``responses`` library, so
these tests run fully offline with no real network access.
"""

import requests
from unittest.mock import patch

import pandas as pd
import pytest
import responses as responses_lib

from openf1 import (
    BASE_URL,
    OpenF1Client,
    OpenF1UnavailableError,
    validate_dataframe,
)


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


# ===========================================================================
# session_status (tri-state) + session_exists (boolean wrapper)
# ===========================================================================

class TestSessionStatus:
    @responses_lib.activate
    def test_valid_key_is_exists(self):
        responses_lib.add(
            responses_lib.GET,
            f"{BASE_URL}/sessions",
            json=[{"session_key": 9472, "session_name": "Race"}],
            status=200,
        )
        client = OpenF1Client(mode="historical")
        assert client.session_status(9472) == "exists"

    @responses_lib.activate
    def test_renumbered_key_404_is_not_found(self):
        # A definitive 404 short-circuits — no retries, no wasted backoff.
        responses_lib.add(
            responses_lib.GET,
            f"{BASE_URL}/sessions",
            json={"detail": "No results found."},
            status=404,
        )
        client = OpenF1Client(mode="historical", timeout=1)
        assert client.session_status(960) == "not_found"

    @responses_lib.activate
    def test_empty_200_is_not_found(self):
        # OpenF1 answered successfully with no rows — a genuine missing key.
        responses_lib.add(
            responses_lib.GET, f"{BASE_URL}/sessions", json=[], status=200,
        )
        client = OpenF1Client(mode="historical")
        assert client.session_status(960) == "not_found"

    @responses_lib.activate
    @patch("openf1.time.sleep")
    def test_transient_5xx_is_unknown_after_retries(self, mock_sleep):
        # A 5xx (or rate-limit) is transient — retried, then reported as
        # "unknown", NOT "not_found", so callers don't blame the key.
        for _ in range(3):
            responses_lib.add(
                responses_lib.GET, f"{BASE_URL}/sessions",
                json={"detail": "Server error"}, status=503,
            )
        client = OpenF1Client(mode="historical", timeout=1)
        assert client.session_status(9472) == "unknown"

    @responses_lib.activate
    @patch("openf1.time.sleep")
    def test_connection_error_is_unknown(self, mock_sleep):
        for _ in range(3):
            responses_lib.add(
                responses_lib.GET, f"{BASE_URL}/sessions",
                body=requests.ConnectionError("boom"),
            )
        client = OpenF1Client(mode="historical", timeout=1)
        assert client.session_status(9472) == "unknown"


class TestSessionExists:
    @responses_lib.activate
    def test_valid_key_returns_true(self):
        responses_lib.add(
            responses_lib.GET,
            f"{BASE_URL}/sessions",
            json=[{"session_key": 9472, "session_name": "Race"}],
            status=200,
        )
        client = OpenF1Client(mode="historical")
        assert client.session_exists(9472) is True

    @responses_lib.activate
    def test_stale_key_404_returns_false(self):
        # Only "exists" is truthy — a 404 (not_found) maps to False.
        responses_lib.add(
            responses_lib.GET,
            f"{BASE_URL}/sessions",
            json={"detail": "No results found."},
            status=404,
        )
        client = OpenF1Client(mode="historical", timeout=1)
        assert client.session_exists(960) is False


# ===========================================================================
# strict mode — exhausted retries must be distinguishable from "no data"
# ===========================================================================

class TestStrictMode:
    """A lenient client collapses every failure into an empty DataFrame, so a
    rate-limited fetch is indistinguishable from a race that genuinely has no
    data. Strict clients (ETL) raise instead.
    """

    def test_defaults_to_lenient(self):
        assert OpenF1Client(mode="historical").strict is False

    @responses_lib.activate
    @patch("openf1.time.sleep")
    def test_strict_raises_when_retries_exhausted(self, mock_sleep):
        for _ in range(3):
            responses_lib.add(
                responses_lib.GET,
                f"{BASE_URL}/laps",
                json={"detail": "Too Many Requests"},
                status=429,
            )
        client = OpenF1Client(mode="historical", timeout=1, strict=True)
        with pytest.raises(OpenF1UnavailableError, match="laps"):
            client.get_laps(session_key=9999)

    @responses_lib.activate
    @patch("openf1.time.sleep")
    def test_lenient_still_returns_empty_when_retries_exhausted(self, mock_sleep):
        for _ in range(3):
            responses_lib.add(
                responses_lib.GET,
                f"{BASE_URL}/laps",
                json={"detail": "Too Many Requests"},
                status=429,
            )
        client = OpenF1Client(mode="historical", timeout=1, strict=False)
        assert client.get_laps(session_key=9999).empty

    @responses_lib.activate
    def test_strict_returns_empty_on_definitive_no_data(self):
        # 200 with an empty body is a real answer: no rows exist. Strict mode
        # must NOT raise here, or genuinely-absent races become hard errors.
        responses_lib.add(
            responses_lib.GET, f"{BASE_URL}/laps", json=[], status=200,
        )
        client = OpenF1Client(mode="historical", strict=True)
        assert client.get_laps(session_key=9999).empty

    @responses_lib.activate
    @patch("openf1.time.sleep")
    def test_strict_recovers_without_raising_if_a_retry_succeeds(self, mock_sleep):
        responses_lib.add(
            responses_lib.GET,
            f"{BASE_URL}/laps",
            json={"detail": "Too Many Requests"},
            status=429,
        )
        responses_lib.add(
            responses_lib.GET,
            f"{BASE_URL}/laps",
            json=[{
                "driver_number": 4, "lap_number": 1, "lap_duration": 90.1,
                "is_pit_out_lap": False, "st_speed": 300, "session_key": 9999,
                "date_start": "2026-04-06T05:01:30",
            }],
            status=200,
        )
        client = OpenF1Client(mode="historical", timeout=1, strict=True)
        assert not client.get_laps(session_key=9999).empty

    @responses_lib.activate
    @patch("openf1.time.sleep")
    def test_strict_get_sessions_raises_on_rate_limited_meetings(self, mock_sleep):
        # The exact production failure: /meetings 429s out, and the caller used
        # to read the empty result as "OpenF1 returned no sessions".
        for _ in range(3):
            responses_lib.add(
                responses_lib.GET,
                f"{BASE_URL}/meetings",
                json={"detail": "Too Many Requests"},
                status=429,
            )
        client = OpenF1Client(mode="historical", timeout=1, strict=True)
        with pytest.raises(OpenF1UnavailableError, match="meetings"):
            client.get_sessions(year=2026, grand_prix="Monaco Grand Prix")

    @responses_lib.activate
    def test_strict_get_sessions_still_empty_for_unknown_race(self):
        responses_lib.add(
            responses_lib.GET, f"{BASE_URL}/meetings", json=[], status=200,
        )
        client = OpenF1Client(mode="historical", strict=True)
        assert client.get_sessions(year=2026, grand_prix="Nonexistent GP").empty

    @responses_lib.activate
    @patch("openf1.time.sleep")
    def test_strict_treats_404_as_definitive_absence_not_an_outage(self, mock_sleep):
        # OpenF1 answers an unknown meeting_name with 404, not 200+[]. Strict
        # mode must read that as "no such race" rather than raising.
        responses_lib.add(
            responses_lib.GET,
            f"{BASE_URL}/meetings",
            json={"detail": "No results found."},
            status=404,
        )
        client = OpenF1Client(mode="historical", timeout=1, strict=True)
        assert client.get_sessions(year=2026, grand_prix="Nonexistent GP").empty
        # Definitive answer — must not burn retries or backoff on it.
        assert mock_sleep.call_count == 0

    @responses_lib.activate
    @patch("openf1.time.sleep")
    def test_404_short_circuits_without_retrying(self, mock_sleep):
        responses_lib.add(
            responses_lib.GET,
            f"{BASE_URL}/laps",
            json={"detail": "No results found."},
            status=404,
        )
        client = OpenF1Client(mode="historical", timeout=1, strict=True)
        assert client.get_laps(session_key=1).empty
        # A single request, not three.
        assert len(responses_lib.calls) == 1
