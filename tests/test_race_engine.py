"""
test_race_engine.py — Unit tests for race_engine.py

Tests cover RaceAnalyser methods and the standalone utility functions,
all driven by small synthetic DataFrames from conftest.py.
"""

import pandas as pd

from race_engine import (
    RaceAnalyser,
    compute_pace_summary,
    rolling_pace,
    stint_degradation,
)


# ===========================================================================
# RaceAnalyser.rolling_pace
# ===========================================================================

class TestRollingPace:
    def test_returns_dataframe(self, analyser):
        result = analyser.rolling_pace()
        assert isinstance(result, pd.DataFrame)

    def test_has_one_column_per_driver(self, analyser, minimal_laps):
        result = analyser.rolling_pace()
        expected_drivers = set(minimal_laps["driver_number"].unique())
        assert set(result.columns) == expected_drivers

    def test_empty_input_returns_empty_dataframe(self, empty_laps, empty_stints, empty_position):
        ra = RaceAnalyser(empty_laps, empty_stints, empty_position)
        assert ra.rolling_pace().empty

    def test_large_window_handled_gracefully(self, analyser):
        # window=10 with only 5 laps — min_periods=1 so all laps produce a value
        result = analyser.rolling_pace(window=10)
        assert not result.empty

    def test_index_name_is_lap_number(self, analyser):
        result = analyser.rolling_pace()
        assert result.index.name == "lap_number"

    def test_values_are_numeric(self, analyser):
        result = analyser.rolling_pace()
        assert result.select_dtypes(include="number").shape == result.shape


# ===========================================================================
# RaceAnalyser.gap_to_leader
# ===========================================================================

class TestGapToLeader:
    def test_returns_dataframe(self, analyser):
        assert isinstance(analyser.gap_to_leader(), pd.DataFrame)

    def test_leader_column_is_all_zeros(self, analyser):
        gap = analyser.gap_to_leader()
        # The driver with minimum cumulative time at the last lap is the leader
        last_lap = gap.index.max()
        leader_col = gap.loc[last_lap].idxmin()
        assert (gap[leader_col] == 0.0).all(), "Leader column must be all zeros"

    def test_all_gaps_are_non_negative(self, analyser):
        gap = analyser.gap_to_leader()
        assert (gap >= 0).all().all()

    def test_columns_match_drivers(self, analyser, minimal_laps):
        gap = analyser.gap_to_leader()
        expected = set(minimal_laps["driver_number"].unique())
        assert set(gap.columns) == expected

    def test_empty_input_returns_empty(self, empty_laps, empty_stints, empty_position):
        ra = RaceAnalyser(empty_laps, empty_stints, empty_position)
        assert ra.gap_to_leader().empty


# ===========================================================================
# RaceAnalyser.detect_undercuts
# ===========================================================================

class TestDetectUndercuts:
    def test_returns_a_list(self, analyser):
        result = analyser.detect_undercuts()
        assert isinstance(result, list)

    def test_empty_stints_returns_empty_list(self, minimal_laps, empty_stints, empty_position):
        ra = RaceAnalyser(minimal_laps, empty_stints, empty_position)
        assert ra.detect_undercuts() == []

    def test_each_event_has_required_keys(self, analyser):
        for event in analyser.detect_undercuts():
            assert "lap" in event
            assert "attacking_driver" in event
            assert "defending_driver" in event
            assert event["type"] in ("undercut", "overcut")


# ===========================================================================
# RaceAnalyser.project_finishing_order
# ===========================================================================

class TestProjectFinishingOrder:
    def test_returns_a_list(self, analyser):
        result = analyser.project_finishing_order(laps_remaining=5)
        assert isinstance(result, list)

    def test_active_drivers_sorted_by_projected_position(self, analyser):
        result = analyser.project_finishing_order(laps_remaining=5)
        active = [r for r in result if r.get("status") == "running"]
        positions = [r["projected_position"] for r in active]
        assert positions == sorted(positions)

    def test_faster_driver_projects_higher(self, analyser):
        # Driver 1 (fastest pace) should project ahead of driver 3 (slowest)
        result = analyser.project_finishing_order(laps_remaining=5)
        active = {r["driver"]: r["projected_position"] for r in result if r.get("status") == "running"}
        if 1 in active and 3 in active:
            assert active[1] < active[3]

    def test_empty_laps_returns_empty_list(self, empty_laps, empty_stints, empty_position):
        ra = RaceAnalyser(empty_laps, empty_stints, empty_position)
        assert ra.project_finishing_order(laps_remaining=5) == []


# ===========================================================================
# RaceAnalyser.pace_summary
# ===========================================================================

class TestPaceSummary:
    def test_returns_dataframe(self, analyser):
        assert isinstance(analyser.pace_summary(), pd.DataFrame)

    def test_has_required_columns(self, analyser):
        result = analyser.pace_summary()
        for col in ("mean_pace", "median_pace", "fastest_lap", "lap_count"):
            assert col in result.columns

    def test_row_count_matches_driver_count(self, analyser, minimal_laps):
        result = analyser.pace_summary()
        assert len(result) == minimal_laps["driver_number"].nunique()


# ===========================================================================
# Standalone: compute_pace_summary
# ===========================================================================

class TestComputePaceSummary:
    def test_sorted_by_median_pace(self, minimal_laps):
        result = compute_pace_summary(minimal_laps)
        medians = list(result["median_pace"])
        assert medians == sorted(medians)

    def test_pit_out_laps_excluded(self):
        laps = pd.DataFrame({
            "driver_number": [1, 1, 1],
            "lap_number": [1, 2, 3],
            "lap_duration": [90.0, 91.0, 130.0],  # lap 3 is an out-lap
            "is_pit_out_lap": [False, False, True],
        })
        result = compute_pace_summary(laps)
        # Out-lap should be excluded, so fastest lap must be < 100 s
        assert float(result.loc[1, "fastest_lap"]) < 100.0

    def test_empty_input_returns_empty(self):
        laps = pd.DataFrame(columns=["driver_number", "lap_number", "lap_duration"])
        assert compute_pace_summary(laps).empty


# ===========================================================================
# Standalone: rolling_pace
# ===========================================================================

class TestRollingPaceStandalone:
    def test_adds_rolling_pace_column(self, minimal_laps):
        result = rolling_pace(minimal_laps, window=2)
        assert "rolling_pace" in result.columns

    def test_first_lap_is_nan_when_min_periods_not_met(self):
        laps = pd.DataFrame({
            "driver_number": [1, 1, 1],
            "lap_number": [1, 2, 3],
            "lap_duration": [90.0, 91.0, 92.0],
        })
        result = rolling_pace(laps, window=2)
        # With min_periods=window=2, first lap has NaN
        assert pd.isna(result.iloc[0]["rolling_pace"])

    def test_rolling_average_is_correct(self):
        laps = pd.DataFrame({
            "driver_number": [1, 1, 1],
            "lap_number": [1, 2, 3],
            "lap_duration": [90.0, 92.0, 94.0],
        })
        result = rolling_pace(laps, window=2)
        # Lap 3 rolling(2) = mean(92, 94) = 93
        assert abs(result.iloc[2]["rolling_pace"] - 93.0) < 1e-9


# ===========================================================================
# Standalone: stint_degradation
# ===========================================================================

class TestStintDegradation:
    def test_empty_stints_returns_empty(self, empty_laps, empty_stints):
        result = stint_degradation(empty_laps, empty_stints)
        assert result.empty

    def test_returns_deg_per_lap_column(self, minimal_laps, minimal_stints):
        result = stint_degradation(minimal_laps, minimal_stints)
        if not result.empty:
            assert "deg_per_lap" in result.columns

    def test_degradation_sign(self, minimal_laps, minimal_stints):
        # Lap times increase slightly (positive degradation expected)
        result = stint_degradation(minimal_laps, minimal_stints)
        if not result.empty:
            # All slopes should be positive (times get slower with tyre age)
            assert (result["deg_per_lap"] > 0).all()
