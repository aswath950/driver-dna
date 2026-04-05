"""
race_engine.py — Race pace analysis, undercut detection, and projections.

Consumes DataFrames produced by OpenF1Client and computes:
  - per-driver pace statistics (mean, median, rolling, tyre-adjusted)
  - stint degradation rates
  - undercut / overcut windows
  - projected finishing positions based on current pace trends
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEG_FACTOR = 0.07          # extra seconds per lap when tyre age > 20
DEG_THRESHOLD_LAPS = 20    # lap count on a compound before degradation kicks in
SC_THRESHOLD = 1.10        # laps slower than 110% of median are treated as SC laps


class RaceAnalyser:
    """
    Stateful analyser that wraps laps, stints, and position DataFrames
    from ``OpenF1Client`` and exposes high-level race insight methods.

    Parameters
    ----------
    laps : pd.DataFrame
        Must contain at least ``driver_number``, ``lap_number``,
        ``lap_duration``, ``is_pit_out_lap``.
    stints : pd.DataFrame
        Must contain at least ``driver_number``, ``stint_number``,
        ``compound``, ``lap_start``, ``lap_end``.
    position : pd.DataFrame
        Must contain at least ``driver_number``, ``position``, ``date``.
    """

    def __init__(
        self,
        laps: pd.DataFrame,
        stints: pd.DataFrame,
        position: pd.DataFrame,
    ) -> None:
        self._laps = laps.copy()
        self._stints = stints.copy()
        self._position = position.copy()

        # Pre-compute clean laps (no pit-out, no SC, positive duration).
        self._clean = self._build_clean_laps()
        # Pre-compute per-lap positions for every driver.
        self._lap_positions = self._build_lap_positions()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_clean_laps(self) -> pd.DataFrame:
        """Remove pit-out laps, null durations, and safety-car laps."""
        df = self._laps.dropna(subset=["lap_duration"]).copy()
        df = df[df["lap_duration"] > 0]
        if "is_pit_out_lap" in df.columns:
            df = df[df["is_pit_out_lap"] != True]  # noqa: E712
        median_pace = df["lap_duration"].median()
        if pd.notna(median_pace) and median_pace > 0:
            df = df[df["lap_duration"] <= median_pace * SC_THRESHOLD]
        return df.sort_values(["driver_number", "lap_number"])

    def _build_lap_positions(self) -> pd.DataFrame:
        """
        Derive a (driver_number, lap_number) → position mapping.

        The position endpoint gives time-stamped rows.  We pick the
        latest position entry per driver per lap.
        """
        pos = self._position.copy()
        if pos.empty:
            return pd.DataFrame(columns=["driver_number", "lap_number", "position"])

        # Merge lap timestamps so we can assign each position row to a lap
        laps_ts = self._laps[["driver_number", "lap_number"]].drop_duplicates()
        # Fallback: build a position-per-lap by picking the last reported
        # position entry whose date <= lap completion.
        if "date" in pos.columns and "date" in self._laps.columns:
            merged = pd.merge_asof(
                pos.sort_values("date"),
                self._laps[["driver_number", "lap_number", "date"]]
                    .drop_duplicates()
                    .sort_values("date"),
                on="date",
                by="driver_number",
                direction="backward",
            )
            merged = merged.dropna(subset=["lap_number"])
            merged = (
                merged
                .sort_values("date")
                .groupby(["driver_number", "lap_number"])
                .last()
                .reset_index()
            )
            result = merged[["driver_number", "lap_number", "position"]]

            # Forward-fill gaps: if a driver is missing position data for
            # some laps, carry the last known position forward.
            all_laps = sorted(self._laps["lap_number"].dropna().unique())
            all_drivers = result["driver_number"].unique()
            full_idx = pd.MultiIndex.from_product(
                [all_drivers, all_laps], names=["driver_number", "lap_number"],
            )
            result = (
                result
                .set_index(["driver_number", "lap_number"])
                .reindex(full_idx)
                .groupby(level="driver_number")
                .ffill()
                .reset_index()
            )
            return result

        # If no usable timestamps, fall back to sequential numbering
        return pd.DataFrame(columns=["driver_number", "lap_number", "position"])

    def _get_position_at_lap(self, driver: int, lap: int) -> float | None:
        """Return the track position of *driver* at *lap*, or None."""
        row = self._lap_positions.loc[
            (self._lap_positions["driver_number"] == driver)
            & (self._lap_positions["lap_number"] == lap)
        ]
        if row.empty:
            return None
        return float(row.iloc[-1]["position"])

    def _current_stint_info(self, driver: int, lap: int) -> dict | None:
        """Return stint metadata for *driver* at *lap*."""
        st = self._stints
        match = st[
            (st["driver_number"] == driver)
            & (st["lap_start"] <= lap)
            & (st["lap_end"] >= lap)
        ]
        if match.empty:
            return None
        row = match.iloc[-1]
        return {
            "compound": row.get("compound"),
            "stint_number": row.get("stint_number"),
            "tyre_age": int(lap - row.get("lap_start", lap)
                            + row.get("tyre_age_at_start", 0)),
        }

    def _pit_laps(self, driver: int) -> list[int]:
        """Return lap numbers where *driver* pitted (start of a new stint)."""
        st = self._stints[self._stints["driver_number"] == driver]
        st = st.sort_values("stint_number")
        # Each stint after the first starts with a pit stop.
        pit_laps = st["lap_start"].iloc[1:].tolist() if len(st) > 1 else []
        return [int(l) for l in pit_laps if pd.notna(l)]

    # ------------------------------------------------------------------
    # 1. Rolling pace
    # ------------------------------------------------------------------

    def rolling_pace(self, window: int = 5) -> pd.DataFrame:
        """
        Rolling-average lap time per driver (ignoring pit & SC laps).

        Returns
        -------
        DataFrame with ``lap_number`` as index, one column per
        ``driver_number``, values = rolling average lap time (seconds).
        """
        if self._clean.empty:
            return pd.DataFrame()
        df = self._clean.copy()
        frames = {}
        for drv, grp in df.groupby("driver_number"):
            s = (
                grp.set_index("lap_number")["lap_duration"]
                .rolling(window, min_periods=1)
                .mean()
            )
            frames[drv] = s
        result = pd.DataFrame(frames)
        result.index.name = "lap_number"
        return result.sort_index()

    # ------------------------------------------------------------------
    # 2. Gap to leader
    # ------------------------------------------------------------------

    def gap_to_leader(self) -> pd.DataFrame:
        """
        Cumulative time gap to the race leader at each lap.

        Returns
        -------
        DataFrame with ``lap_number`` as index, one column per
        ``driver_number``, values = gap in seconds (leader = 0.0).
        """
        df = self._laps.dropna(subset=["lap_duration"]).copy()
        df = df[df["lap_duration"] > 0]
        if df.empty:
            return pd.DataFrame()
        df = df.sort_values(["driver_number", "lap_number"])

        # Cumulative race time per driver
        df["cum_time"] = df.groupby("driver_number")["lap_duration"].cumsum()
        pivot = df.pivot_table(
            index="lap_number",
            columns="driver_number",
            values="cum_time",
            aggfunc="last",
        )
        # Gap = driver's cum_time minus the leader's cum_time at each lap
        leader_time = pivot.min(axis=1)
        gap = pivot.sub(leader_time, axis=0)
        gap.index.name = "lap_number"
        return gap.sort_index()

    # ------------------------------------------------------------------
    # 3. Undercut / overcut detection
    # ------------------------------------------------------------------

    def detect_undercuts(self, lookahead: int = 3) -> list[dict]:
        """
        Detect undercuts and overcuts based on position changes around
        pit stops.

        An **undercut** is flagged when:
          - Driver A pits on lap N
          - Driver B stays out
          - Before the stop A was behind B
          - Within *lookahead* laps of A rejoining, A is ahead of B

        An **overcut** is the inverse (B stays out longer and comes out
        ahead of A after A pitted first).

        Returns
        -------
        list of dicts, each with keys: ``lap``, ``attacking_driver``,
        ``defending_driver``, ``type`` (``'undercut'`` | ``'overcut'``).
        """
        if self._stints.empty or self._lap_positions.empty:
            return []
        drivers = self._stints["driver_number"].unique()
        events: list[dict] = []
        seen: set[tuple] = set()

        for drv_a in drivers:
            pit_laps_a = self._pit_laps(drv_a)
            for pit_lap in pit_laps_a:
                pos_before = self._get_position_at_lap(drv_a, pit_lap - 1)
                if pos_before is None:
                    continue

                for drv_b in drivers:
                    if drv_b == drv_a:
                        continue
                    pos_b_before = self._get_position_at_lap(drv_b, pit_lap - 1)
                    if pos_b_before is None:
                        continue

                    # Check B didn't also pit on the same lap
                    if pit_lap in self._pit_laps(drv_b):
                        continue

                    a_was_behind = pos_before > pos_b_before

                    # Look ahead for a position swap
                    for check_lap in range(pit_lap + 1, pit_lap + lookahead + 1):
                        pos_a_after = self._get_position_at_lap(drv_a, check_lap)
                        pos_b_after = self._get_position_at_lap(drv_b, check_lap)
                        if pos_a_after is None or pos_b_after is None:
                            continue

                        a_now_ahead = pos_a_after < pos_b_after

                        key = (pit_lap, drv_a, drv_b)
                        if key in seen:
                            break

                        if a_was_behind and a_now_ahead:
                            events.append({
                                "lap": pit_lap,
                                "attacking_driver": drv_a,
                                "defending_driver": drv_b,
                                "type": "undercut",
                            })
                            seen.add(key)
                            break
                        elif not a_was_behind and not a_now_ahead:
                            events.append({
                                "lap": pit_lap,
                                "attacking_driver": drv_b,
                                "defending_driver": drv_a,
                                "type": "overcut",
                            })
                            seen.add(key)
                            break

        return events

    # ------------------------------------------------------------------
    # 4. Project finishing order
    # ------------------------------------------------------------------

    def project_finishing_order(
        self,
        laps_remaining: int,
        window: int = 5,
    ) -> list[dict]:
        """
        Project the final race order accounting for current pace,
        cumulative gaps, and tyre degradation.

        Degradation factor: ``+0.07 s/lap`` for every lap beyond 20 on the
        same compound.

        Returns
        -------
        Sorted list of dicts, each with keys: ``driver``,
        ``projected_total_time``, ``current_position``,
        ``projected_position``, ``position_change``.
        """
        clean = self._clean
        all_laps = self._laps.dropna(subset=["lap_duration"])
        all_laps = all_laps[all_laps["lap_duration"] > 0]
        if all_laps.empty:
            return []

        # Determine the last 3 recorded laps in the session for DNF detection
        session_max_lap = int(all_laps["lap_number"].max()) if not all_laps.empty else 0
        dnf_cutoff = max(1, session_max_lap - 2)  # last 3 laps of session

        active_projections = []
        dnf_projections = []

        for drv, grp in all_laps.groupby("driver_number"):
            grp = grp.sort_values("lap_number")
            elapsed = float(grp["lap_duration"].sum())
            current_lap = int(grp["lap_number"].max())
            current_pos = self._get_position_at_lap(drv, current_lap)

            # DNF check: driver has no laps in the last 3 recorded laps
            if current_lap < dnf_cutoff:
                dnf_projections.append({
                    "driver": drv,
                    "projected_total_time": float("inf"),
                    "current_position": int(current_pos) if current_pos else None,
                    "projected_position": None,
                    "position_change": None,
                    "status": "DNF",
                })
                continue

            # Recent pace from clean laps only
            drv_clean = clean[clean["driver_number"] == drv].sort_values("lap_number")
            recent = drv_clean.tail(window)["lap_duration"]
            base_pace = float(recent.mean()) if len(recent) > 0 else float("inf")

            # Tyre degradation estimate
            stint_info = self._current_stint_info(drv, current_lap)
            current_tyre_age = stint_info["tyre_age"] if stint_info else 0

            projected_remaining = 0.0
            for i in range(1, laps_remaining + 1):
                lap_tyre_age = current_tyre_age + i
                deg = (
                    DEG_FACTOR * (lap_tyre_age - DEG_THRESHOLD_LAPS)
                    if lap_tyre_age > DEG_THRESHOLD_LAPS
                    else 0.0
                )
                projected_remaining += base_pace + deg

            active_projections.append({
                "driver": drv,
                "projected_total_time": round(elapsed + projected_remaining, 3),
                "current_position": int(current_pos) if current_pos else None,
                "status": "running",
            })

        # Sort active drivers by projected total time
        active_projections.sort(key=lambda p: p["projected_total_time"])
        for i, p in enumerate(active_projections, 1):
            p["projected_position"] = i
            curr = p["current_position"]
            p["position_change"] = (curr - i) if curr is not None else None

        # DNF drivers go at the bottom, ordered by the lap they stopped
        dnf_start = len(active_projections) + 1
        for i, p in enumerate(dnf_projections):
            p["projected_position"] = dnf_start + i

        return active_projections + dnf_projections


# ------------------------------------------------------------------
# Standalone utility functions (kept for backward compatibility)
# ------------------------------------------------------------------

def compute_pace_summary(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Per-driver aggregate pace from a laps DataFrame.

    Returns
    -------
    DataFrame indexed by ``driver_number`` with columns:
        mean_pace, median_pace, std_pace, fastest_lap, lap_count
    """
    valid = laps.dropna(subset=["lap_duration"]).copy()
    valid = valid[valid["lap_duration"] > 0]
    if "is_pit_out_lap" in valid.columns:
        valid = valid[valid["is_pit_out_lap"] != True]  # noqa: E712

    grouped = valid.groupby("driver_number")["lap_duration"]
    summary = pd.DataFrame({
        "mean_pace": grouped.mean(),
        "median_pace": grouped.median(),
        "std_pace": grouped.std(),
        "fastest_lap": grouped.min(),
        "lap_count": grouped.count(),
    })
    return summary.sort_values("median_pace")


def rolling_pace(
    laps: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """
    Compute a rolling-average lap time per driver.

    Returns
    -------
    Copy of *laps* with an added ``rolling_pace`` column (NaN for the
    first *window - 1* laps of each driver).
    """
    df = laps.dropna(subset=["lap_duration"]).copy()
    df = df.sort_values(["driver_number", "lap_number"])
    df["rolling_pace"] = (
        df.groupby("driver_number")["lap_duration"]
        .transform(lambda s: s.rolling(window, min_periods=window).mean())
    )
    return df


# ------------------------------------------------------------------
# Stint / tyre analysis
# ------------------------------------------------------------------

def stint_degradation(
    laps: pd.DataFrame,
    stints: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate tyre degradation (seconds per lap) for each stint.

    Fits a simple linear regression (lap_number vs lap_duration) within
    each stint.  Returns one row per stint with the slope as ``deg_per_lap``.
    """
    merged = _merge_stint_info(laps, stints)
    if merged.empty:
        return pd.DataFrame()

    results = []
    for (drv, stint_no), grp in merged.groupby(["driver_number", "stint_number"]):
        grp = grp.dropna(subset=["lap_duration"]).sort_values("lap_number")
        if len(grp) < 3:
            continue
        x = grp["lap_number"].values.astype(float)
        y = grp["lap_duration"].values.astype(float)
        slope, _ = np.polyfit(x, y, 1)
        results.append({
            "driver_number": drv,
            "stint_number": stint_no,
            "compound": grp["compound"].iloc[0] if "compound" in grp.columns else None,
            "laps_in_stint": len(grp),
            "deg_per_lap": round(float(slope), 4),
            "mean_pace": round(float(y.mean()), 3),
        })
    return pd.DataFrame(results)


def _merge_stint_info(
    laps: pd.DataFrame,
    stints: pd.DataFrame,
) -> pd.DataFrame:
    """Attach stint metadata (compound, stint_number) to each lap row."""
    if stints.empty or laps.empty:
        return pd.DataFrame()

    records = []
    for _, stint in stints.iterrows():
        drv = stint["driver_number"]
        lap_start = stint.get("lap_start")
        lap_end = stint.get("lap_end")
        if pd.isna(lap_start) or pd.isna(lap_end):
            continue
        mask = (
            (laps["driver_number"] == drv)
            & (laps["lap_number"] >= lap_start)
            & (laps["lap_number"] <= lap_end)
        )
        chunk = laps.loc[mask].copy()
        chunk["stint_number"] = stint.get("stint_number")
        chunk["compound"] = stint.get("compound")
        records.append(chunk)

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


# ------------------------------------------------------------------
# Undercut / overcut detection
# ------------------------------------------------------------------

def detect_undercut_windows(
    laps: pd.DataFrame,
    stints: pd.DataFrame,
    threshold_seconds: float = 1.5,
) -> pd.DataFrame:
    """
    Identify laps where a driver could gain a net undercut advantage.

    An undercut window exists when:
      - a driver is within *threshold_seconds* of the car ahead on pace, AND
      - the car ahead is on older tyres (higher tyre age).

    Returns
    -------
    DataFrame with columns: ``driver_number``, ``target_driver``,
    ``lap_number``, ``pace_delta``, ``tyre_age_delta``.
    """
    merged = _merge_stint_info(laps, stints)
    if merged.empty:
        return pd.DataFrame()

    merged = merged.dropna(subset=["lap_duration"])
    merged = merged.sort_values(["lap_number", "lap_duration"])

    windows = []
    for lap_no, grp in merged.groupby("lap_number"):
        if len(grp) < 2:
            continue
        grp = grp.sort_values("lap_duration").reset_index(drop=True)
        for i in range(1, len(grp)):
            behind = grp.iloc[i]
            ahead = grp.iloc[i - 1]
            pace_delta = behind["lap_duration"] - ahead["lap_duration"]
            tyre_age_behind = behind.get("tyre_age", 0) or 0
            tyre_age_ahead = ahead.get("tyre_age", 0) or 0
            tyre_age_delta = tyre_age_ahead - tyre_age_behind

            if abs(pace_delta) <= threshold_seconds and tyre_age_delta > 0:
                windows.append({
                    "driver_number": behind["driver_number"],
                    "target_driver": ahead["driver_number"],
                    "lap_number": lap_no,
                    "pace_delta": round(float(pace_delta), 3),
                    "tyre_age_delta": int(tyre_age_delta),
                })
    return pd.DataFrame(windows)


# ------------------------------------------------------------------
# Projected finishing order
# ------------------------------------------------------------------

def project_finishing_order(
    laps: pd.DataFrame,
    position: pd.DataFrame,
    total_laps: int,
    recent_window: int = 5,
) -> pd.DataFrame:
    """
    Project the final race order based on each driver's recent pace trend.

    For each driver the most recent *recent_window* laps are averaged.
    Remaining laps are multiplied by that pace to estimate total remaining
    time.  Drivers are ranked by total projected race time.

    Returns
    -------
    DataFrame sorted by projected finishing position with columns:
        ``driver_number``, ``laps_completed``, ``recent_pace``,
        ``projected_remaining``, ``projected_total``, ``projected_position``.
    """
    valid = laps.dropna(subset=["lap_duration"]).copy()
    valid = valid[valid["lap_duration"] > 0]
    valid = valid.sort_values(["driver_number", "lap_number"])

    projections = []
    for drv, grp in valid.groupby("driver_number"):
        completed = int(grp["lap_number"].max())
        remaining = max(0, total_laps - completed)
        recent = grp.tail(recent_window)["lap_duration"]
        recent_pace = float(recent.mean()) if len(recent) > 0 else float("inf")
        elapsed = float(grp["lap_duration"].sum())
        projected_remaining = recent_pace * remaining
        projections.append({
            "driver_number": drv,
            "laps_completed": completed,
            "recent_pace": round(recent_pace, 3),
            "projected_remaining": round(projected_remaining, 3),
            "projected_total": round(elapsed + projected_remaining, 3),
        })

    proj_df = pd.DataFrame(projections).sort_values("projected_total")
    proj_df["projected_position"] = range(1, len(proj_df) + 1)
    return proj_df.reset_index(drop=True)
