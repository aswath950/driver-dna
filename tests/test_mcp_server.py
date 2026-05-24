"""
tests/test_mcp_server.py — Unit tests for mcp/server.py

All OpenF1 network calls and telemetry fetch functions are mocked so
tests run fully offline. Tests cover all four MCP tools, helper utilities,
and error-path behaviour.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

# Ensure src/ and mcp/ are importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "mcp"))

import server as mcp_server  # the MCP server module under test


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def sessions_df() -> pd.DataFrame:
    """Two-session weekend: Qualifying (key=1001) and Race (key=1002)."""
    return pd.DataFrame([
        {
            "session_key": 1001, "session_name": "Qualifying",
            "session_type": "Qualifying",
            "date_start": pd.Timestamp("2024-09-07 14:00:00", tz="UTC"),
        },
        {
            "session_key": 1002, "session_name": "Race",
            "session_type": "Race",
            "date_start": pd.Timestamp("2024-09-08 13:00:00", tz="UTC"),
        },
    ])


@pytest.fixture
def drivers_df() -> pd.DataFrame:
    """Two drivers with distinct team colours."""
    return pd.DataFrame([
        {
            "driver_number": 1, "name_acronym": "VER",
            "full_name": "Max Verstappen", "team_name": "Red Bull Racing",
            "team_colour": "#3671C6",
        },
        {
            "driver_number": 44, "name_acronym": "HAM",
            "full_name": "Lewis Hamilton", "team_name": "Mercedes",
            "team_colour": "#27F4D2",
        },
    ])


@pytest.fixture
def laps_df() -> pd.DataFrame:
    """Two drivers × two laps each, clean (no pit-out laps)."""
    base = pd.Timestamp("2024-09-07 14:10:00", tz="UTC")
    return pd.DataFrame([
        {"driver_number": 1,  "lap_number": 10, "lap_duration": 80.5,
         "is_pit_out_lap": False, "date_start": base},
        {"driver_number": 1,  "lap_number": 11, "lap_duration": 79.8,
         "is_pit_out_lap": False, "date_start": base + pd.Timedelta(seconds=80.5)},
        {"driver_number": 44, "lap_number": 10, "lap_duration": 80.9,
         "is_pit_out_lap": False, "date_start": base},
        {"driver_number": 44, "lap_number": 11, "lap_duration": 80.1,
         "is_pit_out_lap": False, "date_start": base + pd.Timedelta(seconds=80.9)},
    ])


@pytest.fixture
def telemetry_data() -> dict:
    """Synthetic full telemetry dict (200-point arrays) for one driver."""
    t = np.linspace(0.0, 1.0, 200)
    return {
        "speed":    t * 300.0,
        "throttle": t * 100.0,
        "brake":    (1.0 - t) * 100.0,
        "cumtime":  t * 79.8,
        "x": None, "y": None,
        "lap_time": 79.8, "lap_number": 11,
    }


@pytest.fixture
def circuit_xy() -> dict:
    """Minimal circuit coordinate data for Track Map tests."""
    return {"x": list(range(500)), "y": list(range(500, 1000))}


# ── Helper utilities ──────────────────────────────────────────────────────────

class TestToList:
    def test_converts_numpy_array_to_plain_floats(self):
        arr = np.array([1.0, 2.5, 3.9])
        result = mcp_server._to_list(arr)
        assert result == [1.0, 2.5, 3.9]
        assert all(isinstance(v, float) for v in result)

    def test_returns_empty_list_for_none(self):
        assert mcp_server._to_list(None) == []

    def test_casts_numpy_float64_to_python_float(self):
        arr = np.array([np.float64(42.0)])
        result = mcp_server._to_list(arr)
        assert isinstance(result[0], float)
        assert not isinstance(result[0], np.floating)


class TestAcronymMap:
    def test_maps_driver_number_to_acronym(self, drivers_df):
        result = mcp_server._acronym_map(drivers_df)
        assert result == {1: "VER", 44: "HAM"}

    def test_handles_missing_acronym_as_unk(self):
        df = pd.DataFrame([{"driver_number": 5, "name_acronym": pd.NA}])
        result = mcp_server._acronym_map(df)
        assert result[5] == "UNK"

    def test_skips_rows_with_null_driver_number(self):
        df = pd.DataFrame([{"driver_number": pd.NA, "name_acronym": "TST"}])
        result = mcp_server._acronym_map(df)
        assert result == {}


class TestColourMap:
    def test_returns_hex_with_hash_prefix(self):
        df = pd.DataFrame([
            {"driver_number": 1,  "team_colour": "3671C6"},
            {"driver_number": 44, "team_colour": "#27F4D2"},
        ])
        result = mcp_server._colour_map(df)
        assert result[1]  == "#3671C6"
        assert result[44] == "#27F4D2"

    def test_returns_empty_dict_when_column_absent(self, drivers_df):
        df = drivers_df.drop(columns=["team_colour"])
        assert mcp_server._colour_map(df) == {}

    def test_skips_null_colour_rows(self):
        df = pd.DataFrame([
            {"driver_number": 1,  "team_colour": pd.NA},
            {"driver_number": 44, "team_colour": "#FF0000"},
        ])
        result = mcp_server._colour_map(df)
        assert 1 not in result
        assert result[44] == "#FF0000"


class TestResolveSession:
    def test_returns_matching_session_key(self, sessions_df):
        with patch("server.OpenF1Client") as MockClient:
            MockClient.return_value.get_sessions.return_value = sessions_df
            key, _ = mcp_server._resolve_session(2024, "Italian Grand Prix", "Race")
            assert key == 1002

    def test_matching_is_case_insensitive(self, sessions_df):
        with patch("server.OpenF1Client") as MockClient:
            MockClient.return_value.get_sessions.return_value = sessions_df
            key, _ = mcp_server._resolve_session(2024, "Italian Grand Prix", "qualifying")
            assert key == 1001

    def test_raises_on_empty_sessions_response(self):
        with patch("server.OpenF1Client") as MockClient:
            MockClient.return_value.get_sessions.return_value = pd.DataFrame()
            with pytest.raises(ValueError, match="No sessions found"):
                mcp_server._resolve_session(2024, "Unknown GP", "Race")

    def test_raises_when_session_type_absent(self, sessions_df):
        with patch("server.OpenF1Client") as MockClient:
            MockClient.return_value.get_sessions.return_value = sessions_df
            with pytest.raises(ValueError, match="not found"):
                mcp_server._resolve_session(2024, "Italian Grand Prix", "Sprint")

    def test_error_message_lists_available_sessions(self, sessions_df):
        with patch("server.OpenF1Client") as MockClient:
            MockClient.return_value.get_sessions.return_value = sessions_df
            with pytest.raises(ValueError, match="Qualifying"):
                mcp_server._resolve_session(2024, "Italian Grand Prix", "FP1")


# ── list_sessions ─────────────────────────────────────────────────────────────

class TestListSessions:
    def test_returns_all_sessions(self, sessions_df):
        with patch("server.OpenF1Client") as MockClient:
            MockClient.return_value.get_sessions.return_value = sessions_df
            result = mcp_server.list_sessions(2024, "Italian Grand Prix")
            assert len(result) == 2

    def test_session_fields_are_present(self, sessions_df):
        with patch("server.OpenF1Client") as MockClient:
            MockClient.return_value.get_sessions.return_value = sessions_df
            result = mcp_server.list_sessions(2024, "Italian Grand Prix")
            assert result[0]["session_type"] == "Qualifying"
            assert result[1]["session_key"] == 1002

    def test_timestamps_serialised_as_iso_strings(self, sessions_df):
        with patch("server.OpenF1Client") as MockClient:
            MockClient.return_value.get_sessions.return_value = sessions_df
            result = mcp_server.list_sessions(2024, "Italian Grand Prix")
            assert isinstance(result[0]["date_start"], str)
            assert "2024-09-07" in result[0]["date_start"]

    def test_returns_empty_list_for_unknown_race(self):
        with patch("server.OpenF1Client") as MockClient:
            MockClient.return_value.get_sessions.return_value = pd.DataFrame()
            result = mcp_server.list_sessions(2024, "Unknown Race")
            assert result == []


# ── list_drivers ──────────────────────────────────────────────────────────────

class TestListDrivers:
    def test_returns_all_drivers(self, sessions_df, drivers_df):
        with patch("server.OpenF1Client") as MockClient:
            inst = MockClient.return_value
            inst.get_sessions.return_value = sessions_df
            inst.get_drivers.return_value = drivers_df
            result = mcp_server.list_drivers(2024, "Italian Grand Prix", "Race")
            assert len(result) == 2

    def test_driver_fields_are_correct(self, sessions_df, drivers_df):
        with patch("server.OpenF1Client") as MockClient:
            inst = MockClient.return_value
            inst.get_sessions.return_value = sessions_df
            inst.get_drivers.return_value = drivers_df
            result = mcp_server.list_drivers(2024, "Italian Grand Prix", "Race")
            nums = {r["driver_number"] for r in result}
            acrs = {r["name_acronym"] for r in result}
            assert nums == {1, 44}
            assert acrs == {"VER", "HAM"}

    def test_raises_for_unknown_session_type(self, sessions_df):
        with patch("server.OpenF1Client") as MockClient:
            MockClient.return_value.get_sessions.return_value = sessions_df
            with pytest.raises(ValueError):
                mcp_server.list_drivers(2024, "Italian Grand Prix", "Sprint")

    def test_returns_empty_list_when_no_drivers(self, sessions_df):
        with patch("server.OpenF1Client") as MockClient:
            inst = MockClient.return_value
            inst.get_sessions.return_value = sessions_df
            inst.get_drivers.return_value = pd.DataFrame()
            result = mcp_server.list_drivers(2024, "Italian Grand Prix", "Race")
            assert result == []


# ── get_fastest_lap_data ──────────────────────────────────────────────────────

class TestGetFastestLapData:
    def _setup_client(self, MockClient, sessions_df, laps_df, drivers_df):
        inst = MockClient.return_value
        inst.get_sessions.return_value = sessions_df
        inst.get_laps.return_value = laps_df
        inst.get_drivers.return_value = drivers_df

    def test_returns_all_telemetry_channels(
        self, sessions_df, drivers_df, laps_df, telemetry_data
    ):
        with patch("server.OpenF1Client") as MockClient, \
             patch("server._fetch_fastest_lap_all_openf1", return_value=telemetry_data):
            self._setup_client(MockClient, sessions_df, laps_df, drivers_df)
            result = mcp_server.get_fastest_lap_data(
                2024, "Italian Grand Prix", "Race", 1
            )
            assert result["driver_number"] == 1
            assert result["acronym"] == "VER"
            assert result["lap_time"] == 79.8
            assert result["lap_number"] == 11
            for ch in ("speed", "throttle", "brake", "cumtime"):
                assert len(result[ch]) == 200

    def test_arrays_contain_plain_python_floats(
        self, sessions_df, drivers_df, laps_df, telemetry_data
    ):
        with patch("server.OpenF1Client") as MockClient, \
             patch("server._fetch_fastest_lap_all_openf1", return_value=telemetry_data):
            self._setup_client(MockClient, sessions_df, laps_df, drivers_df)
            result = mcp_server.get_fastest_lap_data(
                2024, "Italian Grand Prix", "Race", 1
            )
            assert all(isinstance(v, float) for v in result["speed"])
            assert not isinstance(result["speed"][0], np.floating)

    def test_raises_when_telemetry_unavailable(
        self, sessions_df, drivers_df, laps_df
    ):
        with patch("server.OpenF1Client") as MockClient, \
             patch("server._fetch_fastest_lap_all_openf1", return_value=None):
            self._setup_client(MockClient, sessions_df, laps_df, drivers_df)
            with pytest.raises(ValueError, match="Could not fetch telemetry"):
                mcp_server.get_fastest_lap_data(
                    2024, "Italian Grand Prix", "Race", 99
                )

    def test_unknown_driver_acronym_falls_back_to_unk(
        self, sessions_df, drivers_df, laps_df, telemetry_data
    ):
        with patch("server.OpenF1Client") as MockClient, \
             patch("server._fetch_fastest_lap_all_openf1", return_value=telemetry_data):
            self._setup_client(MockClient, sessions_df, laps_df, drivers_df)
            # Driver 77 is not in drivers_df
            result = mcp_server.get_fastest_lap_data(
                2024, "Italian Grand Prix", "Race", 77
            )
            assert result["acronym"] == "UNK"


# ── get_channel_comparison ────────────────────────────────────────────────────

class TestGetChannelComparison:
    def _setup_client(self, MockClient, sessions_df, laps_df, drivers_df):
        inst = MockClient.return_value
        inst.get_sessions.return_value = sessions_df
        inst.get_laps.return_value = laps_df
        inst.get_drivers.return_value = drivers_df

    def test_invalid_channel_raises_immediately(
        self, sessions_df, drivers_df, laps_df
    ):
        with patch("server.OpenF1Client") as MockClient:
            self._setup_client(MockClient, sessions_df, laps_df, drivers_df)
            with pytest.raises(ValueError, match="Invalid channel"):
                mcp_server.get_channel_comparison(
                    2024, "Italian Grand Prix", "Race", 1, 44, "DRS"
                )

    def test_speed_channel_returns_speed_arrays_and_figure(
        self, sessions_df, drivers_df, laps_df
    ):
        trace = np.linspace(100.0, 300.0, 200)
        with patch("server.OpenF1Client") as MockClient, \
             patch("server._fetch_fastest_lap_openf1", return_value=(trace, 79.8, 11)):
            self._setup_client(MockClient, sessions_df, laps_df, drivers_df)
            result = mcp_server.get_channel_comparison(
                2024, "Italian Grand Prix", "Race", 1, 44, "Speed"
            )
            assert result["channel"] == "Speed"
            assert len(result["driver_a"]["speed"]) == 200
            assert len(result["driver_b"]["speed"]) == 200
            assert isinstance(result["figure_json"], str)
            fig_dict = json.loads(result["figure_json"])
            assert "data" in fig_dict

    def test_throttle_channel_stores_throttle_key(
        self, sessions_df, drivers_df, laps_df
    ):
        trace = np.linspace(0.0, 100.0, 200)
        with patch("server.OpenF1Client") as MockClient, \
             patch("server._fetch_fastest_lap_openf1", return_value=(trace, 79.8, 11)):
            self._setup_client(MockClient, sessions_df, laps_df, drivers_df)
            result = mcp_server.get_channel_comparison(
                2024, "Italian Grand Prix", "Race", 1, 44, "Throttle"
            )
            assert "throttle" in result["driver_a"]
            assert "throttle" in result["driver_b"]

    def test_brake_channel_stores_brake_key(
        self, sessions_df, drivers_df, laps_df
    ):
        trace = np.zeros(200)
        with patch("server.OpenF1Client") as MockClient, \
             patch("server._fetch_fastest_lap_openf1", return_value=(trace, 79.8, 11)):
            self._setup_client(MockClient, sessions_df, laps_df, drivers_df)
            result = mcp_server.get_channel_comparison(
                2024, "Italian Grand Prix", "Race", 1, 44, "Brake"
            )
            assert "brake" in result["driver_a"]

    def test_time_delta_calls_build_time_delta_fig(
        self, sessions_df, drivers_df, laps_df, telemetry_data
    ):
        mock_fig = go.Figure()
        with patch("server.OpenF1Client") as MockClient, \
             patch("server._fetch_fastest_lap_all_openf1", return_value=telemetry_data), \
             patch("server._build_time_delta_fig", return_value=mock_fig) as mock_td:
            self._setup_client(MockClient, sessions_df, laps_df, drivers_df)
            result = mcp_server.get_channel_comparison(
                2024, "Italian Grand Prix", "Race", 1, 44, "Time Delta"
            )
            assert mock_td.call_count == 1
            assert result["channel"] == "Time Delta"
            assert isinstance(result["figure_json"], str)

    def test_time_delta_response_includes_all_channels(
        self, sessions_df, drivers_df, laps_df, telemetry_data
    ):
        mock_fig = go.Figure()
        with patch("server.OpenF1Client") as MockClient, \
             patch("server._fetch_fastest_lap_all_openf1", return_value=telemetry_data), \
             patch("server._build_time_delta_fig", return_value=mock_fig):
            self._setup_client(MockClient, sessions_df, laps_df, drivers_df)
            result = mcp_server.get_channel_comparison(
                2024, "Italian Grand Prix", "Race", 1, 44, "Time Delta"
            )
            for ch in ("speed", "throttle", "brake", "cumtime"):
                assert ch in result["driver_a"]
                assert ch in result["driver_b"]

    def test_track_map_injects_circuit_xy(
        self, sessions_df, drivers_df, laps_df, telemetry_data, circuit_xy
    ):
        mock_fig = go.Figure()
        with patch("server.OpenF1Client") as MockClient, \
             patch("server._fetch_fastest_lap_all_openf1", return_value=telemetry_data), \
             patch("server._build_track_map_fig", return_value=mock_fig) as mock_tm, \
             patch.dict("server.CIRCUITS", {"Italian Grand Prix": circuit_xy}):
            self._setup_client(MockClient, sessions_df, laps_df, drivers_df)
            result = mcp_server.get_channel_comparison(
                2024, "Italian Grand Prix", "Race", 1, 44, "Track Map"
            )
            assert mock_tm.call_count == 1
            assert result["channel"] == "Track Map"
            # Verify XY was injected into the data dicts passed to the figure builder
            call_kwargs = mock_tm.call_args
            data_a_arg = call_kwargs[0][0]
            assert data_a_arg["x"] == circuit_xy["x"]

    def test_track_map_raises_when_circuit_not_in_circuits_dict(
        self, sessions_df, drivers_df, laps_df, telemetry_data
    ):
        with patch("server.OpenF1Client") as MockClient, \
             patch("server._fetch_fastest_lap_all_openf1", return_value=telemetry_data), \
             patch("server._build_track_map_fig", return_value=None), \
             patch.dict("server.CIRCUITS", {}, clear=True):
            self._setup_client(MockClient, sessions_df, laps_df, drivers_df)
            with pytest.raises(ValueError, match="Track map unavailable"):
                mcp_server.get_channel_comparison(
                    2024, "Italian Grand Prix", "Race", 1, 44, "Track Map"
                )

    def test_raises_when_driver_a_has_no_telemetry(
        self, sessions_df, drivers_df, laps_df, telemetry_data
    ):
        def side_effect(session_key, driver_number, laps_df):
            return None if driver_number == 1 else telemetry_data

        with patch("server.OpenF1Client") as MockClient, \
             patch("server._fetch_fastest_lap_all_openf1", side_effect=side_effect):
            self._setup_client(MockClient, sessions_df, laps_df, drivers_df)
            with pytest.raises(ValueError, match="No telemetry for driver 1"):
                mcp_server.get_channel_comparison(
                    2024, "Italian Grand Prix", "Race", 1, 44, "Time Delta"
                )

    def test_raises_when_driver_b_has_no_telemetry(
        self, sessions_df, drivers_df, laps_df, telemetry_data
    ):
        def side_effect(session_key, driver_number, laps_df):
            return telemetry_data if driver_number == 1 else None

        with patch("server.OpenF1Client") as MockClient, \
             patch("server._fetch_fastest_lap_all_openf1", side_effect=side_effect):
            self._setup_client(MockClient, sessions_df, laps_df, drivers_df)
            with pytest.raises(ValueError, match="No telemetry for driver 44"):
                mcp_server.get_channel_comparison(
                    2024, "Italian Grand Prix", "Race", 1, 44, "Time Delta"
                )

    def test_figure_json_is_valid_plotly_json(
        self, sessions_df, drivers_df, laps_df
    ):
        trace = np.linspace(200.0, 350.0, 200)
        with patch("server.OpenF1Client") as MockClient, \
             patch("server._fetch_fastest_lap_openf1", return_value=(trace, 79.8, 11)):
            self._setup_client(MockClient, sessions_df, laps_df, drivers_df)
            result = mcp_server.get_channel_comparison(
                2024, "Italian Grand Prix", "Race", 1, 44, "Speed"
            )
            fig_dict = json.loads(result["figure_json"])
            assert "data" in fig_dict
            assert "layout" in fig_dict
            assert len(fig_dict["data"]) == 2  # one trace per driver

    def test_raises_when_channel_trace_unavailable_for_driver_a(
        self, sessions_df, drivers_df, laps_df
    ):
        trace_b = np.linspace(100.0, 300.0, 200)

        def side_effect(session_key, driver_number, laps_df, channel):
            if driver_number == 1:
                return None, None, None
            return trace_b, 80.1, 11

        with patch("server.OpenF1Client") as MockClient, \
             patch("server._fetch_fastest_lap_openf1", side_effect=side_effect):
            self._setup_client(MockClient, sessions_df, laps_df, drivers_df)
            with pytest.raises(ValueError, match="No Speed telemetry for driver 1"):
                mcp_server.get_channel_comparison(
                    2024, "Italian Grand Prix", "Race", 1, 44, "Speed"
                )
