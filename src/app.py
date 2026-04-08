"""
app.py — Streamlit dashboard for Driver DNA Fingerprinter.

Run with:
    streamlit run src/app.py

Expected columns from dataset.parquet:
  driver, lap_time_seconds, mean_speed, max_speed, min_speed,
  throttle_mean, throttle_std, brake_mean, brake_events,
  gear_changes, steer_std, speed_trace, throttle_trace, brake_trace
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent))

from model import load_model, FEATURE_COLS, DATA_DIR, MODELS_DIR
from openf1 import OpenF1Client
from race_engine import RaceAnalyser

DATASET_PATH = DATA_DIR / "dataset.parquet"
CIRCUITS_PATH = DATA_DIR / "circuits.json"
ACCURACY_PATH = MODELS_DIR / "accuracy.txt"
N_POINTS = 200
TRACE_COLS = {"Speed": "speed_trace", "Throttle": "throttle_trace", "Brake": "brake_trace"}
# OpenF1 car_data column names for the same channels
OPENF1_TRACE_COLS = {"Speed": "speed", "Throttle": "throttle", "Brake": "brake"}
# Channels that require special rendering (not a simple 1D line overlay)
SPECIAL_CHANNELS = ["Track Map", "Time Delta"]
RADAR_FEATURES = ["mean_speed", "throttle_mean", "throttle_std", "brake_events", "steer_std", "gear_changes"]

# --- Session state defaults for non-widget keys only ---------------
# Widget keys (ta_a, ta_b, radar_drivers, mystery_lap, rp_mode, etc.)
# must NOT be pre-set here — Streamlit manages widget state internally
# and raises a conflict error if a widget key already exists in
# session_state when the widget renders with its own default value.
STATE_DEFAULTS: dict = {
    # Tab 4 — non-widget state
    "rp_last_refresh": None,
    "rp_live_client": None,
    # Tab 4 historical — persists loaded race data across reruns
    "rp_hist_analyser": None,
    "rp_hist_driver_map": {},
    "rp_hist_team_colour_map": {},   # {int(driver_number): "#RRGGBB"}
    "rp_hist_session_key": None,     # int session_key of the currently loaded session
    # Tab 4 historical — available sessions fetched from OpenF1
    "rp_available_sessions": None,   # pd.DataFrame or None
    "rp_fetched_for": None,          # (year, gp) tuple — invalidates cache on change
    # Tab 4 live — driver map for current session
    "rp_live_driver_map": {},
    "rp_live_team_colour_map": {},   # {int(driver_number): "#RRGGBB"}
    "rp_live_meeting_name": "",      # GP name for live Track Map circuit lookup
}

st.set_page_config(page_title="Driver DNA Fingerprinter", layout="wide")

for _key, _val in STATE_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val

# --- Sidebar ---
with st.sidebar:
    st.title("🧬 Driver DNA + 🏁 Race Pace")

    # -- Driver DNA section --
    st.subheader("Driver DNA")
    st.markdown("Identify F1 drivers from telemetry alone using XGBoost.")
    if ACCURACY_PATH.exists():
        acc = float(ACCURACY_PATH.read_text().strip())
        st.metric("Model CV Accuracy", f"{acc:.1%}")
    else:
        st.caption("Run `model.py` to generate accuracy.txt")
    st.info("**Mystery Driver** — pick any lap and see if the ML model can identify the driver from driving style alone.")

    st.divider()

    # -- Race Dashboard section --
    st.subheader("Race Dashboard")
    rp_mode_val = st.session_state.get("rp_mode", "")
    if rp_mode_val.startswith("📡"):
        st.markdown("**Mode:** 📡 Live")
        last_ts = st.session_state.get("rp_last_refresh")
        if last_ts:
            st.caption(f"Last refreshed: {last_ts}")
        else:
            st.caption("Waiting for first refresh...")
    elif rp_mode_val.startswith("🗂️"):
        st.markdown("**Mode:** 🗂️ Historical")
        loaded_gp = st.session_state.get("rp_gp", "—")
        loaded_year = st.session_state.get("rp_year", "—")
        st.caption(f"{loaded_year} {loaded_gp}")
    else:
        st.caption("Select a mode in the Race Dashboard tab to get started.")
    st.info("**Race Dashboard** — fastest lap telemetry comparison, plus live or historical race analysis with rolling pace, gap charts, undercut detection, and projected finishing order.")

    # -- Health Check expander --
    st.divider()
    with st.expander("🩺 Health Check"):
        _ok = "✅"
        _fail = "❌"

        # Dataset
        st.markdown(f"{_ok if DATASET_PATH.exists() else _fail} `dataset.parquet` exists")

        # Model
        _model_exists = (MODELS_DIR / "driver_dna_clf.joblib").exists()
        st.markdown(f"{_ok if _model_exists else _fail} `driver_dna_clf.joblib` exists")

        # OpenF1 API reachable (cached for 60s so it doesn't block every rerun)
        @st.cache_data(ttl=60, show_spinner=False)
        def _check_openf1() -> bool:
            try:
                import requests as _req
                _resp = _req.get(
                    "https://api.openf1.org/v1/sessions", timeout=5, params={"limit": 1}
                )
                return _resp.status_code == 200
            except Exception:
                return False

        _api_ok = _check_openf1()
        st.markdown(f"{_ok if _api_ok else _fail} OpenF1 API reachable")

        # Required libraries
        _libs = ["fastf1", "pandas", "numpy", "xgboost", "sklearn", "shap",
                 "streamlit", "plotly", "joblib", "matplotlib", "requests"]
        _missing_libs = []
        for _lib in _libs:
            try:
                __import__(_lib)
            except ImportError:
                _missing_libs.append(_lib)
        if _missing_libs:
            st.markdown(f"{_fail} Missing libraries: {', '.join(_missing_libs)}")
        else:
            st.markdown(f"{_ok} All required libraries importable")

# --- Load resources ---
@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_parquet(DATASET_PATH)


@st.cache_data
def get_circuits() -> dict:
    """Load pre-generated circuit XY outlines keyed by Grand Prix name."""
    if not CIRCUITS_PATH.exists():
        return {}
    with open(CIRCUITS_PATH) as f:
        return json.load(f)


if not DATASET_PATH.exists():
    st.error("Run `python src/pipeline.py` first to fetch telemetry and build the dataset.")
    st.stop()

model_path = MODELS_DIR / "driver_dna_clf.joblib"
if not model_path.exists():
    st.error("Run `python src/model.py` first to train the model.")
    st.stop()

try:
    clf, le = get_model()
    df = get_data()
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()

circuits = get_circuits()

drivers = sorted(df["driver"].unique())
palette = px.colors.qualitative.Plotly
color_map = {d: palette[i % len(palette)] for i, d in enumerate(drivers)}

tab1, tab2, tab3 = st.tabs(["Driver Radar", "Mystery Driver", "Race Dashboard"])

# ================================================================
# TAB 1 — Driver Radar
# ================================================================
with tab1:
    st.subheader("Driver Style Fingerprint")

    selected = st.multiselect(
        "Select 2–4 drivers",
        options=drivers,
        default=drivers[:min(3, len(drivers))],
        key="radar_drivers",
    )

    if len(selected) < 2:
        st.warning("Select at least 2 drivers.")
    elif len(selected) > 4:
        st.warning("Select at most 4 drivers.")
    else:
        feat_df = df[RADAR_FEATURES].copy()
        feat_min = feat_df.min()
        feat_max = feat_df.max()
        norm_df = (feat_df - feat_min) / (feat_max - feat_min).replace(0, 1)
        norm_df["driver"] = df["driver"].values

        driver_means = norm_df.groupby("driver")[RADAR_FEATURES].mean()

        categories = RADAR_FEATURES + [RADAR_FEATURES[0]]
        fig = go.Figure()
        for drv in selected:
            vals = driver_means.loc[drv].tolist()
            vals += [vals[0]]
            fig.add_trace(go.Scatterpolar(
                r=vals,
                theta=categories,
                fill="toself",
                name=drv,
                line=dict(color=color_map[drv]),
                opacity=0.7,
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Driver Style Radar (normalised 0–1)",
            legend_title="Driver",
        )
        st.plotly_chart(fig, width="stretch", key="tab2_driver_radar")

# ================================================================
# TAB 2 — Mystery Driver
# ================================================================
with tab2:
    st.subheader("Can the model identify the driver?")

    lap_idx = st.selectbox(
        "Pick a lap",
        range(len(df)),
        format_func=lambda i: f"Lap {i + 1}",
        key="mystery_lap",
    )

    if st.button("Identify Driver", key="identify_btn"):
        lap_row = df.iloc[lap_idx]
        actual_driver = lap_row["driver"]

        X_input = lap_row[FEATURE_COLS].to_numpy().reshape(1, -1)
        proba = clf.predict_proba(X_input)[0]
        pred_driver = le.classes_[proba.argmax()]

        prob_df = pd.DataFrame({
            "Driver": le.classes_,
            "Probability": proba,
        }).sort_values("Probability", ascending=True)

        fig = go.Figure(go.Bar(
            x=prob_df["Probability"],
            y=prob_df["Driver"],
            orientation="h",
            marker_color=[color_map.get(d, "#888") for d in prob_df["Driver"]],
        ))
        fig.update_layout(
            title="Model Prediction Probabilities",
            xaxis_title="Probability",
            yaxis_title="Driver",
            xaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig, width="stretch", key="tab3_mystery_proba")

        correct = pred_driver == actual_driver
        icon = "✅" if correct else "❌"
        st.markdown(f"### Actual driver: **{actual_driver}** {icon}")
        if correct:
            st.success(f"Correct! The model identified **{pred_driver}** with {proba.max():.1%} confidence.")
        else:
            st.error(f"Wrong! The model predicted **{pred_driver}** but the actual driver was **{actual_driver}**.")

# ================================================================
# TAB 4 — Race Pace Dashboard
# ================================================================

# ---- helpers shared by both modes --------------------------------

RACE_PALETTE = px.colors.qualitative.Dark24


def _fastest_lap_trace(
    telemetry_df: pd.DataFrame,
    driver_acronym: str,
    channel: str,
) -> tuple[np.ndarray | None, float | None, int | None]:
    """
    Extract the fastest lap telemetry trace for a driver from the FastF1 dataset.

    Parameters
    ----------
    telemetry_df : pd.DataFrame
        The FastF1 ``dataset.parquet`` already loaded at startup (global ``df``).
    driver_acronym : str
        3-letter driver code as stored in ``df["driver"]``, e.g. ``"HAM"``.
    channel : str
        One of ``"Speed"``, ``"Throttle"``, ``"Brake"``.

    Returns
    -------
    (trace, lap_time, lap_number) where trace is a float64 ndarray of length N_POINTS,
    or (None, None, None) if the driver or channel is not found.
    lap_number is the driver's own lap number (e.g. lap 32 of their race), or None
    if the dataset pre-dates the lap_number column.

    Note: this function is now used only by the Driver Radar and Mystery Driver tabs,
    which still read from the static dataset.parquet. The Race Dashboard telemetry
    section uses _fetch_fastest_lap_openf1() and _fetch_fastest_lap_all_openf1()
    instead, which pull live data from the OpenF1 API for the user-selected session.
    """
    trace_col = TRACE_COLS.get(channel)
    if trace_col is None or trace_col not in telemetry_df.columns:
        return None, None, None

    drv_df = telemetry_df[telemetry_df["driver"] == driver_acronym].dropna(
        subset=["lap_time_seconds"]
    )
    if drv_df.empty:
        return None, None, None

    fastest_row = drv_df.sort_values("lap_time_seconds").iloc[0]
    lap_time = float(fastest_row["lap_time_seconds"])
    trace = np.asarray(fastest_row[trace_col], dtype=float)
    lap_number: int | None = int(fastest_row["lap_number"]) if "lap_number" in fastest_row.index else None
    return trace, lap_time, lap_number


def _get_fastest_lap_all(
    telemetry_df: pd.DataFrame,
    driver_acronym: str,
) -> dict | None:
    """
    Return all resampled traces for a driver's fastest lap, or None if not found.

    Keys: 'speed', 'x', 'y', 'lap_time', 'lap_number'.
    'x' and 'y' are None if the dataset pre-dates the x_trace/y_trace columns.

    Note: this function is now used only to supply X/Y circuit coordinates to Track Map
    from dataset.parquet (the only available source for XY). All other telemetry in the
    Race Dashboard is fetched live via _fetch_fastest_lap_all_openf1().
    """
    drv_df = telemetry_df[telemetry_df["driver"] == driver_acronym].dropna(
        subset=["lap_time_seconds"]
    )
    if drv_df.empty:
        return None
    row = drv_df.sort_values("lap_time_seconds").iloc[0]
    return {
        "speed": np.asarray(row["speed_trace"], dtype=float),
        "x": np.asarray(row["x_trace"], dtype=float) if "x_trace" in row.index else None,
        "y": np.asarray(row["y_trace"], dtype=float) if "y_trace" in row.index else None,
        "lap_time": float(row["lap_time_seconds"]),
        "lap_number": int(row["lap_number"]) if "lap_number" in row.index else None,
    }


def _fetch_fastest_lap_openf1(
    session_key: int | str,
    driver_number: int,
    laps_df: pd.DataFrame,
    channel: str,
) -> tuple[np.ndarray | None, float | None, int | None]:
    """
    Fetch the fastest-lap telemetry trace for a driver from the OpenF1 car_data API.

    Parameters
    ----------
    session_key : int | str
        The OpenF1 session key for the currently loaded session, or ``"latest"``
        for the live session.
    driver_number : int
        The OpenF1 driver number (e.g. 44 for Hamilton).
    laps_df : pd.DataFrame
        The laps DataFrame already fetched from OpenF1 for this session.
        Used to identify the fastest lap and its time window — avoids a second API call.
    channel : str
        One of ``"Speed"``, ``"Throttle"``, ``"Brake"``.

    Returns
    -------
    (trace, lap_time, lap_number) where trace is a float64 ndarray of length N_POINTS,
    or (None, None, None) on any failure.

    How it works
    ------------
    1. Filter laps to this driver, drop pit-out laps, find the row with the
       minimum ``lap_duration``.
    2. Use ``date_start`` and ``date_start + lap_duration`` as the car_data
       time window (ISO-8601 strings passed to ``date>`` / ``date<``).
    3. Fetch car_data from OpenF1 for that window.
    4. Sort by date, compute a cumulative-distance proxy from speed × dt, then
       resample the requested channel to N_POINTS evenly spaced distance points
       using np.interp — matching the normalisation pipeline.py applies.
    """
    OPENF1_CHANNEL_COL = {"Speed": "speed", "Throttle": "throttle", "Brake": "brake"}
    col = OPENF1_CHANNEL_COL.get(channel)
    if col is None:
        return None, None, None

    # --- 1. Find the fastest lap for this driver ---
    drv_laps = laps_df[laps_df["driver_number"] == driver_number].copy()
    if "is_pit_out_lap" in drv_laps.columns:
        drv_laps = drv_laps[drv_laps["is_pit_out_lap"] != True]  # noqa: E712
    drv_laps = drv_laps.dropna(subset=["lap_duration"])
    if drv_laps.empty:
        return None, None, None

    fastest = drv_laps.sort_values("lap_duration").iloc[0]
    lap_time = float(fastest["lap_duration"])
    lap_number: int | None = int(fastest["lap_number"]) if pd.notna(fastest.get("lap_number")) else None

    # --- 2. Build the time window ---
    date_start = fastest.get("date_start")
    if date_start is None or pd.isna(date_start):
        return None, None, None
    ts_start = pd.Timestamp(date_start)
    ts_end = ts_start + pd.Timedelta(seconds=lap_time)
    # Add a small buffer so we don't clip the final sample
    date_gte = ts_start.isoformat()
    date_lte = (ts_end + pd.Timedelta(seconds=0.5)).isoformat()

    # --- 3. Fetch car_data ---
    client = OpenF1Client(mode="historical")
    car = client.get_car_data(
        session_key=session_key,
        driver_number=driver_number,
        date_gte=date_gte,
        date_lte=date_lte,
    )
    if car.empty or col not in car.columns or "date" not in car.columns:
        return None, None, None

    car = car.sort_values("date").reset_index(drop=True)
    values = car[col].to_numpy(dtype=float)
    if len(values) < 2:
        return None, None, None

    # --- 4. Resample to N_POINTS evenly spaced distance points ---
    # Compute cumulative distance proxy: distance ≈ speed × dt (speed in km/h → m/s)
    dt = car["date"].diff().dt.total_seconds().fillna(0.0).to_numpy(dtype=float)[1:]
    speeds_ms = car["speed"].to_numpy(dtype=float)[:-1] / 3.6            # km/h → m/s
    dist_increments = np.where(np.isfinite(speeds_ms) & np.isfinite(dt), speeds_ms * dt, 0.0)
    dist = np.concatenate([[0.0], np.cumsum(dist_increments)])

    dist_grid = np.linspace(dist[0], dist[-1], N_POINTS)
    trace = np.interp(dist_grid, dist, values)
    return trace, lap_time, lap_number


def _fetch_fastest_lap_all_openf1(
    session_key: int | str,
    driver_number: int,
    laps_df: pd.DataFrame,
) -> dict | None:
    """
    Fetch all telemetry channels for a driver's fastest lap from the OpenF1 API.

    Returns a dict with keys: ``'speed'``, ``'throttle'``, ``'brake'``,
    ``'lap_time'``, ``'lap_number'``, ``'x'``, ``'y'``.

    ``'x'`` and ``'y'`` are always ``None`` — OpenF1 car_data does not include
    circuit XY coordinates. Track Map falls back to dataset.parquet for these.
    """
    drv_laps = laps_df[laps_df["driver_number"] == driver_number].copy()
    if "is_pit_out_lap" in drv_laps.columns:
        drv_laps = drv_laps[drv_laps["is_pit_out_lap"] != True]  # noqa: E712
    drv_laps = drv_laps.dropna(subset=["lap_duration"])
    if drv_laps.empty:
        return None

    fastest = drv_laps.sort_values("lap_duration").iloc[0]
    lap_time = float(fastest["lap_duration"])
    lap_number: int | None = int(fastest["lap_number"]) if pd.notna(fastest.get("lap_number")) else None

    date_start = fastest.get("date_start")
    if date_start is None or pd.isna(date_start):
        return None
    ts_start = pd.Timestamp(date_start)
    ts_end = ts_start + pd.Timedelta(seconds=lap_time)
    date_gte = ts_start.isoformat()
    date_lte = (ts_end + pd.Timedelta(seconds=0.5)).isoformat()

    client = OpenF1Client(mode="historical")
    car = client.get_car_data(
        session_key=session_key,
        driver_number=driver_number,
        date_gte=date_gte,
        date_lte=date_lte,
    )
    if car.empty or "date" not in car.columns or "speed" not in car.columns:
        return None

    car = car.sort_values("date").reset_index(drop=True)
    speeds = car["speed"].to_numpy(dtype=float)
    if len(speeds) < 2:
        return None

    # Cumulative elapsed time from real timestamps — immune to missing samples.
    # (date[i] - date[0]) gives the true wall-clock time at each data point;
    # no speed-based reconstruction needed so there are no integration artifacts.
    cumtime_raw = (car["date"] - car["date"].iloc[0]).dt.total_seconds().to_numpy(dtype=float)

    dt = car["date"].diff().dt.total_seconds().fillna(0.0).to_numpy(dtype=float)[1:]
    speeds_ms = speeds[:-1] / 3.6
    dist_increments = np.where(np.isfinite(speeds_ms) & np.isfinite(dt), speeds_ms * dt, 0.0)
    dist = np.concatenate([[0.0], np.cumsum(dist_increments)])
    dist_grid = np.linspace(dist[0], dist[-1], N_POINTS)

    def _resample(col: str) -> np.ndarray:
        if col not in car.columns:
            return np.full(N_POINTS, np.nan)
        return np.interp(dist_grid, dist, car[col].to_numpy(dtype=float))

    return {
        "speed": _resample("speed"),
        "throttle": _resample("throttle"),
        "brake": _resample("brake"),
        "cumtime": np.interp(dist_grid, dist, cumtime_raw),  # elapsed seconds at each distance point
        "x": None,   # not available from OpenF1 car_data; Track Map uses dataset.parquet
        "y": None,
        "lap_time": lap_time,
        "lap_number": lap_number,
    }


def _build_track_map_fig(
    data_a: dict, acronym_a: str, color_a: str,
    data_b: dict, acronym_b: str, color_b: str,
) -> "go.Figure | None":
    """
    Circuit coloured by the faster driver per microsector.

    Each of the N-1 distance gaps is coloured by whichever driver had the higher
    speed at that point. Consecutive same-winner segments are merged into one
    Scatter trace to keep the figure lightweight.
    Returns None if X/Y coordinates are missing from the dataset.
    """
    x_a, y_a, spd_a = data_a["x"], data_a["y"], data_a["speed"]
    spd_b = data_b["speed"]

    if x_a is None or y_a is None:
        return None

    # Upsample to 1000 display points for a smoother circuit outline and finer
    # microsector boundaries. XY (from circuits.json, 500 pts) and speed traces
    # (from OpenF1, N_POINTS=200) may have different source lengths, so each is
    # interpolated on its own unit-interval grid before being mapped to t_fine.
    N_DISPLAY = 1000
    t_fine = np.linspace(0.0, 1.0, N_DISPLAY)

    t_xy = np.linspace(0.0, 1.0, len(x_a))
    x_fine = np.interp(t_fine, t_xy, x_a)
    y_fine = np.interp(t_fine, t_xy, y_a)

    t_spd = np.linspace(0.0, 1.0, len(spd_a))
    spd_a_fine = np.interp(t_fine, t_spd, spd_a)
    spd_b_fine = np.interp(t_fine, t_spd, spd_b)

    # winner[i] = "a" or "b" for the microsector between point i and i+1
    winner = np.where(spd_a_fine[:-1] >= spd_b_fine[:-1], "a", "b")

    fig = go.Figure()

    # Grey background track (full circuit, behind the coloured segments)
    fig.add_trace(go.Scatter(
        x=x_fine, y=y_fine,
        mode="lines",
        line=dict(color="rgba(120,120,120,0.25)", width=10),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Group consecutive same-winner segments and draw each as a single trace
    i = 0
    n = len(winner)
    legend_added: dict[str, bool] = {"a": False, "b": False}
    while i < n:
        w = winner[i]
        j = i
        while j < n and winner[j] == w:
            j += 1
        seg_x = x_fine[i: j + 1]
        seg_y = y_fine[i: j + 1]
        col = color_a if w == "a" else color_b
        label = acronym_a if w == "a" else acronym_b
        fig.add_trace(go.Scatter(
            x=seg_x, y=seg_y,
            mode="lines",
            line=dict(color=col, width=5),
            name=label,
            showlegend=not legend_added[w],
            legendgroup=w,
            hoverinfo="skip",
        ))
        legend_added[w] = True
        i = j

    fig.update_layout(
        title=f"Track Map — {acronym_a} vs {acronym_b} (faster driver by microsector)",
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        legend_title="Faster driver",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=40, b=0),
        height=520,
    )
    return fig


def _build_time_delta_fig(
    data_a: dict, acronym_a: str, color_a: str,
    data_b: dict, acronym_b: str, color_b: str,
) -> go.Figure:
    """
    Redesigned time-delta chart.

    X axis: Lap Distance (0–100%) — symmetric, matches F1 broadcast convention.
    Y axis: cumulative gap in seconds (positive = A ahead, negative = B ahead).

    Delta is computed from speed traces reconstructed to cumulative time, then
    differenced. The line is coloured by whichever driver is ahead at each point.
    Crossover markers, peak advantage markers, and a final-gap annotation are added
    so the chart is readable at a glance.
    """
    def _to_rgba(color: str, alpha: float = 0.30) -> str:
        """Convert any hex or rgb() colour to an rgba() string Plotly accepts."""
        if color.startswith("rgba("):
            return color
        if color.startswith("rgb("):
            return color.replace("rgb(", "rgba(").replace(")", f",{alpha})")
        h = color.lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    # Use real timestamp-derived cumulative time — avoids speed-reconstruction artifacts
    # that cause mid-lap delta spikes when car_data samples are unevenly distributed.
    cumtime_a = data_a["cumtime"]
    cumtime_b = data_b["cumtime"]
    delta = cumtime_b - cumtime_a   # positive = A gaining

    n = len(delta)
    x_pct = np.linspace(0.0, 100.0, n)

    # Per-point plain-language hover text
    hover_text = [
        f"{acronym_a} +{abs(d):.3f}s ahead" if d >= 0 else f"{acronym_b} +{abs(d):.3f}s ahead"
        for d in delta
    ]

    fig = go.Figure()

    # --- Shaded fill regions (opacity 0.30) ---
    pos = np.where(delta >= 0, delta, 0.0)
    neg = np.where(delta < 0, delta, 0.0)

    fig.add_trace(go.Scatter(
        x=np.concatenate([[x_pct[0]], x_pct, [x_pct[-1]]]),
        y=np.concatenate([[0], pos, [0]]),
        mode="lines",
        fill="tozeroy",
        fillcolor=_to_rgba(color_a, 0.30),
        line=dict(width=0),
        name=f"{acronym_a} faster",
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([[x_pct[0]], x_pct, [x_pct[-1]]]),
        y=np.concatenate([[0], neg, [0]]),
        mode="lines",
        fill="tozeroy",
        fillcolor=_to_rgba(color_b, 0.30),
        line=dict(width=0),
        name=f"{acronym_b} faster",
        showlegend=False,
        hoverinfo="skip",
    ))

    # --- Coloured delta line: A's colour above zero, B's colour below ---
    delta_a = np.where(delta >= 0, delta, np.nan)
    delta_b = np.where(delta < 0,  delta, np.nan)

    fig.add_trace(go.Scatter(
        x=x_pct, y=delta_a,
        mode="lines",
        connectgaps=False,
        line=dict(color=color_a, width=2.5),
        name=f"{acronym_a} faster",
        customdata=hover_text,
        hovertemplate="%{x:.0f}%  —  %{customdata}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x_pct, y=delta_b,
        mode="lines",
        connectgaps=False,
        line=dict(color=color_b, width=2.5),
        name=f"{acronym_b} faster",
        customdata=hover_text,
        hovertemplate="%{x:.0f}%  —  %{customdata}<extra></extra>",
    ))

    # --- Lead-change markers ---
    crossings = np.where(np.diff(np.sign(delta)) != 0)[0]
    if len(crossings) > 0:
        cx = x_pct[crossings]
        fig.add_trace(go.Scatter(
            x=cx, y=np.zeros(len(cx)),
            mode="markers",
            marker=dict(color="white", size=8, symbol="circle",
                        line=dict(color="rgba(0,0,0,0.6)", width=1)),
            name="Lead change",
            hovertemplate="Lead change at %{x:.0f}%<extra></extra>",
        ))

    # --- Peak advantage markers (only if meaningful gap > 0.05s) ---
    max_delta = float(np.nanmax(delta))
    min_delta = float(np.nanmin(delta))

    if max_delta > 0.05:
        idx_max = int(np.argmax(delta))
        fig.add_trace(go.Scatter(
            x=[x_pct[idx_max]], y=[max_delta],
            mode="markers",
            marker=dict(color=color_a, size=10, symbol="triangle-up",
                        line=dict(color="white", width=1)),
            name=f"Peak {acronym_a}",
            hovertemplate=f"Max {acronym_a} +{max_delta:.3f}s at %{{x:.0f}}%<extra></extra>",
        ))

    if min_delta < -0.05:
        idx_min = int(np.argmin(delta))
        fig.add_trace(go.Scatter(
            x=[x_pct[idx_min]], y=[min_delta],
            mode="markers",
            marker=dict(color=color_b, size=10, symbol="triangle-down",
                        line=dict(color="white", width=1)),
            name=f"Peak {acronym_b}",
            hovertemplate=f"Max {acronym_b} +{abs(min_delta):.3f}s at %{{x:.0f}}%<extra></extra>",
        ))

    # --- Final gap annotation at right edge ---
    final_gap = float(delta[-1])
    if abs(final_gap) < 0.001:
        gap_text = "Dead heat"
        gap_color = "white"
    elif final_gap > 0:
        gap_text = f"+{final_gap:.3f}s\n{acronym_a}"
        gap_color = color_a
    else:
        gap_text = f"{final_gap:.3f}s\n{acronym_b}"
        gap_color = color_b

    fig.add_annotation(
        x=x_pct[-1], y=final_gap,
        text=gap_text,
        showarrow=True,
        arrowhead=2,
        arrowcolor=gap_color,
        arrowwidth=1.5,
        ax=40, ay=0,
        font=dict(color=gap_color, size=12),
        xanchor="left",
    )

    # --- Title with subtitle ---
    lap_a_num = data_a.get("lap_number")
    lap_b_num = data_b.get("lap_number")
    lap_a_str = f"Lap {lap_a_num}, {data_a['lap_time']:.3f}s" if lap_a_num else f"{data_a['lap_time']:.3f}s"
    lap_b_str = f"Lap {lap_b_num}, {data_b['lap_time']:.3f}s" if lap_b_num else f"{data_b['lap_time']:.3f}s"
    title_text = (
        f"Lap Time Delta — {acronym_a} vs {acronym_b}"
        f"<br><sup>{acronym_a}: {lap_a_str}  |  {acronym_b}: {lap_b_str}</sup>"
    )

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14)),
        xaxis=dict(
            title="Lap Distance (%)",
            range=[0, 100],
            zeroline=False,
        ),
        yaxis=dict(
            title=f"Gap (s)   ↑ {acronym_a} faster   ·   {acronym_b} faster ↓",
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.6)",
            zerolinewidth=1.5,
        ),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=460,
        margin=dict(t=80, b=50, l=60, r=120),
    )
    return fig


def _driver_color_map(
    driver_numbers: list,
    team_colour_map: dict | None = None,
) -> dict:
    """
    Assign a unique colour to each driver number.

    Prefers the team colour from *team_colour_map* when available.
    If two drivers share the same team colour (teammates), the second one
    falls back to the next unused Dark24 colour so every driver is distinct.
    """
    result: dict = {}
    used_colours: set[str] = set()
    palette_idx = 0
    for d in sorted(driver_numbers):
        team_col = team_colour_map.get(d) if team_colour_map else None
        if team_col and team_col not in used_colours:
            result[d] = team_col
            used_colours.add(team_col)
        else:
            # No team colour, or teammate collision — pick next unused Dark24 colour
            while RACE_PALETTE[palette_idx % len(RACE_PALETTE)] in used_colours:
                palette_idx += 1
            col = RACE_PALETTE[palette_idx % len(RACE_PALETTE)]
            result[d] = col
            used_colours.add(col)
            palette_idx += 1
    return result


def _resolve_pair_colours(
    drv_a: int,
    drv_b: int,
    team_colour_map: dict | None,
) -> tuple[str, str]:
    """
    Return a pair of distinct colours for a two-driver telemetry comparison.

    Rules:
    - Driver A always gets their team colour (falls back to Dark24[0] if unknown).
    - Driver B gets their team colour when it differs from Driver A's.
    - If both drivers share the same team colour (teammates), Driver B gets
      the next Dark24 colour that is not already used by Driver A.
    """
    tc = team_colour_map or {}
    col_a = tc.get(drv_a) or RACE_PALETTE[0]
    col_b_candidate = tc.get(drv_b)

    if col_b_candidate and col_b_candidate != col_a:
        # Different teams — use both team colours directly
        return col_a, col_b_candidate

    # Same team (or Driver B has no team colour) — find an unused Dark24 colour for B
    for candidate in RACE_PALETTE:
        if candidate != col_a:
            return col_a, candidate

    # Unreachable in practice (Dark24 has 24 colours), but safe fallback
    return col_a, "#888888"


@st.cache_data(ttl=30, show_spinner=False)
def _cached_analysis(laps_hash: str, _analyser: RaceAnalyser, laps_remaining: int) -> dict:
    """Cache the 4 analysis results so reruns don't recompute.

    *laps_hash* is a hashable key derived from the lap data so the cache
    invalidates when new laps arrive.  *_analyser* is prefixed with ``_``
    so Streamlit skips hashing it (it's not picklable).
    """
    return {
        "rolling_pace": _analyser.rolling_pace(window=5),
        "gap_to_leader": _analyser.gap_to_leader(),
        "undercuts": _analyser.detect_undercuts(),
        "projections": _analyser.project_finishing_order(laps_remaining),
        "driver_numbers": sorted(_analyser._clean["driver_number"].unique()),
        "tyre_deg": _analyser.tyre_degradation(),
        "pace_summary": _analyser.pace_summary(),
    }


def _render_charts(
    analyser: RaceAnalyser,
    laps_remaining: int,
    container,
    driver_map: dict | None = None,
    selected_drivers: list | None = None,
    team_colour_map: dict | None = None,
) -> None:
    """Render all Race-Pace charts inside *container*.

    Parameters
    ----------
    driver_map : dict, optional
        Mapping of ``{driver_number: acronym}`` e.g. ``{44: 'HAM'}``.
        Falls back to the raw driver number string when not provided or
        when a number has no entry in the map.
    selected_drivers : list, optional
        Subset of driver numbers to display. Defaults to all drivers.
    team_colour_map : dict, optional
        Mapping of ``{driver_number: "#RRGGBB"}`` for team colours.
        Drivers with a unique team colour use it; teammates fall back to Dark24.
    """
    if driver_map is None:
        driver_map = {}

    def _label(num) -> str:
        """Return acronym if known, otherwise the number as a string."""
        return driver_map.get(int(num), str(num)) if driver_map else str(num)

    # Build a hash key from the laps data so the cache invalidates on new data
    laps_hash = str(hash(tuple(analyser._laps["lap_number"].tolist()[:50])))
    results = _cached_analysis(laps_hash, analyser, laps_remaining)

    # Resolve the active driver set (filter or all)
    active_drivers = set(selected_drivers) if selected_drivers else set(results["driver_numbers"])
    cmap = _driver_color_map(list(active_drivers), team_colour_map=team_colour_map)

    # ---- Chart 1: Rolling Race Pace --------------------------------
    container.markdown("### Rolling Race Pace")
    rp = results["rolling_pace"]
    fig1 = go.Figure()
    for drv in [d for d in rp.columns if d in active_drivers]:
        fig1.add_trace(go.Scatter(
            x=rp.index, y=rp[drv],
            mode="lines", name=_label(drv),
            line=dict(color=cmap.get(drv, "#888"), width=2),
        ))
    # Session median pace (dashed reference)
    all_vals = rp.values[~np.isnan(rp.values)]
    if len(all_vals):
        median_pace = float(np.median(all_vals))
        fig1.add_hline(
            y=median_pace, line_dash="dash", line_color="white",
            opacity=0.5,
            annotation_text=f"Median {median_pace:.2f}s",
            annotation_position="top left",
        )
    fig1.update_layout(
        xaxis_title="Lap Number", yaxis_title="Rolling Avg Lap Time (s)",
        hovermode="x unified", legend_title="Driver",
    )
    container.plotly_chart(fig1, width="stretch", key="tab4_rolling_pace")

    # ---- Chart 2 & 3: Gap to Leader + Undercut markers -------------
    container.markdown("### Gap to Leader")
    gap = results["gap_to_leader"]
    undercuts = results["undercuts"]

    fig2 = go.Figure()
    for drv in [d for d in gap.columns if d in active_drivers]:
        fig2.add_trace(go.Scatter(
            x=gap.index, y=gap[drv],
            mode="lines", name=_label(drv),
            line=dict(color=cmap.get(drv, "#888"), width=2),
            fill="tozeroy", opacity=0.3,
        ))
    # Overlay undercut/overcut markers (only when both drivers are selected)
    visible_undercuts = [
        e for e in undercuts
        if e["attacking_driver"] in active_drivers and e["defending_driver"] in active_drivers
    ]
    for evt in visible_undercuts:
        lap = evt["lap"]
        atk = evt["attacking_driver"]
        kind = evt["type"]
        marker_symbol = "triangle-up" if kind == "undercut" else "triangle-down"
        try:
            y_val = gap.loc[lap, atk].item()  # type: ignore[union-attr]
        except (KeyError, AttributeError):
            y_val = 0.0
        label = f"Lap {lap}: {_label(atk)} {kind} {_label(evt['defending_driver'])}"
        fig2.add_trace(go.Scatter(
            x=[lap], y=[y_val],
            mode="markers+text",
            marker=dict(
                symbol=marker_symbol, size=14,
                color="lime" if kind == "undercut" else "orange",
                line=dict(width=1, color="white"),
            ),
            text=[f"{'▲' if kind == 'undercut' else '▽'}"],
            textposition="top center",
            hovertext=[label],
            hoverinfo="text",
            showlegend=False,
        ))
    fig2.update_layout(
        xaxis_title="Lap Number", yaxis_title="Gap to Leader (s)",
        hovermode="x unified", legend_title="Driver",
    )
    container.plotly_chart(fig2, width="stretch", key="tab4_gap_to_leader")

    # ---- Chart 4: Projected Finishing Order -------------------------
    container.markdown("### Projected Finishing Order")
    projections = [p for p in results["projections"] if p["driver"] in active_drivers]
    if projections:
        # Winner time comes from the first active (non-DNF) driver
        active = [p for p in projections if p.get("status") != "DNF"]
        winner_time = active[0]["projected_total_time"] if active else 0
        proj_data = []
        for p in projections:
            is_dnf = p.get("status") == "DNF"
            gap_to_win = 0.0 if is_dnf else p["projected_total_time"] - winner_time
            change = p.get("position_change")
            if is_dnf:
                bar_color = "#555555"
            elif change is not None and change > 0:
                bar_color = "#2ecc71"
            elif change is not None and change < 0:
                bar_color = "#e74c3c"
            else:
                bar_color = "#95a5a6"
            cur = p.get("current_position", "?")
            proj_pos = p["projected_position"]
            label = "DNF" if is_dnf else f"P{cur} → P{proj_pos}"
            proj_data.append({
                "driver": _label(p["driver"]),
                "gap": round(gap_to_win, 2),
                "color": bar_color,
                "label": label,
            })
        # Reverse so P1 is at the top of the horizontal bar chart
        proj_data = proj_data[::-1]
        fig4 = go.Figure(go.Bar(
            x=[d["gap"] for d in proj_data],
            y=[d["driver"] for d in proj_data],
            orientation="h",
            marker_color=[d["color"] for d in proj_data],
            text=[d["label"] for d in proj_data],
            textposition="outside",
            hovertemplate="Driver %{y}<br>Gap: +%{x:.2f}s<extra></extra>",
        ))
        fig4.update_layout(
            xaxis_title="Projected Gap to Winner (s)",
            yaxis_title="Driver",
            showlegend=False,
        )
        container.plotly_chart(fig4, width="stretch", key="tab4_projected_order")
    else:
        container.info("Not enough data to project finishing order.")

    # ---- Chart 5: Average Race Pace --------------------------------
    container.markdown("### Average Race Pace")
    pace_sum = results["pace_summary"]
    if pace_sum.empty:
        container.info("Not enough clean lap data for pace summary.")
    else:
        pace_sum = pace_sum.reset_index()
        pace_sum = pace_sum[pace_sum["driver_number"].isin(active_drivers)]
        pace_sum["label"] = pace_sum["driver_number"].apply(_label)
        pace_sum = pace_sum.sort_values("median_pace", ascending=False)  # fastest at top
        fig5 = go.Figure(go.Bar(
            x=pace_sum["median_pace"],
            y=pace_sum["label"],
            orientation="h",
            marker=dict(
                color=pace_sum["std_pace"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Pace Std Dev (s)"),
            ),
            customdata=pace_sum[["mean_pace", "std_pace", "fastest_lap", "lap_count"]].values,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Median: %{x:.3f}s<br>"
                "Mean: %{customdata[0]:.3f}s<br>"
                "Std Dev: %{customdata[1]:.3f}s<br>"
                "Fastest lap: %{customdata[2]:.3f}s<br>"
                "Laps counted: %{customdata[3]}<extra></extra>"
            ),
        ))
        fig5.update_layout(
            xaxis_title="Median Lap Time (s)",
            yaxis_title="Driver",
            showlegend=False,
        )
        container.plotly_chart(fig5, width="stretch", key="tab4_avg_pace")

    # ---- Chart 6: Race Pace Ranking --------------------------------
    container.markdown("### Race Pace Ranking")
    if pace_sum.empty:
        container.info("Not enough clean lap data for pace ranking.")
    else:
        ranked = pace_sum.sort_values("mean_pace")
        bar_colors = [cmap.get(int(row["driver_number"]), "#888") for _, row in ranked.iterrows()]
        fig6 = go.Figure(go.Bar(
            x=ranked["label"],
            y=ranked["mean_pace"],
            error_y=dict(type="data", array=ranked["std_pace"].tolist(), visible=True),
            marker_color=bar_colors,
            text=ranked["fastest_lap"].round(3).astype(str) + "s",
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Mean pace: %{y:.3f}s<br>"
                "Fastest lap: %{text}<extra></extra>"
            ),
        ))
        fig6.update_layout(
            xaxis_title="Driver",
            yaxis_title="Mean Lap Time (s)",
            showlegend=False,
        )
        container.plotly_chart(fig6, width="stretch", key="tab4_pace_ranking")

    # ---- Chart 7: Tyre Degradation by Compound ---------------------
    container.markdown("### Tyre Degradation by Compound")
    tyre_deg = results["tyre_deg"]
    tyre_deg = tyre_deg[tyre_deg["driver_number"].isin(active_drivers)]
    if tyre_deg.empty:
        container.info("Stint data unavailable — tyre degradation chart requires stint information.")
    else:
        COMPOUND_COLORS = {
            "SOFT": "#e8002d",
            "MEDIUM": "#ffd700",
            "HARD": "#cccccc",
            "INTERMEDIATE": "#39b54a",
            "WET": "#0067ff",
        }
        fig7 = go.Figure()
        for compound, grp in tyre_deg.groupby("compound"):
            color = COMPOUND_COLORS.get(str(compound).upper(), "#888888")
            fig7.add_trace(go.Scatter(
                x=grp["mean_pace"],
                y=grp["deg_per_lap"],
                mode="markers",
                name=str(compound),
                marker=dict(color=color, size=10, line=dict(width=1, color="white")),
                customdata=grp[["driver_number", "laps_in_stint"]].assign(
                    label=grp["driver_number"].apply(_label)
                )[["label", "laps_in_stint"]].values,
                hovertemplate=(
                    "<b>%{customdata[0]}</b> — " + str(compound) + "<br>"
                    "Mean pace: %{x:.3f}s<br>"
                    "Degradation: %{y:+.4f}s/lap<br>"
                    "Laps in stint: %{customdata[1]}<extra></extra>"
                ),
            ))
        fig7.add_hline(
            y=0, line_dash="dash", line_color="white",
            opacity=0.4, annotation_text="No degradation",
            annotation_position="top left",
        )
        fig7.update_layout(
            xaxis_title="Mean Stint Pace (s)",
            yaxis_title="Degradation (s / lap)",
            legend_title="Compound",
        )
        container.plotly_chart(fig7, width="stretch", key="tab4_tyre_deg")


# ---- Tab 3 content -----------------------------------------------

with tab3:
    st.subheader("Race Dashboard")

    # ---- Race Analysis ---------------------------------------------
    mode = st.radio(
        "Mode",
        ["🗂️ Historical (select a race)", "📡 Live (current race weekend)"],
        horizontal=True,
        key="rp_mode",
    )

    if mode.startswith("🗂️"):
        # ---- HISTORICAL MODE ----
        h1, h2 = st.columns(2)
        hist_year = h1.selectbox("Year", [2026, 2025, 2024, 2023, 2022], key="rp_year")
        _gp_options = [
            "Australian Grand Prix",
            "Bahrain Grand Prix",
            "Saudi Arabian Grand Prix",
            "Japanese Grand Prix",
            "Chinese Grand Prix",
            "Miami Grand Prix",
            "Emilia Romagna Grand Prix",
            "Monaco Grand Prix",
            "Canadian Grand Prix",
            "Spanish Grand Prix",
            "Austrian Grand Prix",
            "British Grand Prix",
            "Hungarian Grand Prix",
            "Belgian Grand Prix",
            "Dutch Grand Prix",
            "Italian Grand Prix",
            "Azerbaijan Grand Prix",
            "Singapore Grand Prix",
            "United States Grand Prix",
            "Mexico City Grand Prix",
            "São Paulo Grand Prix",
            "Las Vegas Grand Prix",
            "Qatar Grand Prix",
            "Abu Dhabi Grand Prix",
        ]
        hist_gp = h2.selectbox(
            "Grand Prix",
            options=_gp_options,
            index=_gp_options.index("Italian Grand Prix"),
            key="rp_gp",
        )
        # Invalidate cached sessions when year or GP changes
        if st.session_state["rp_fetched_for"] != (hist_year, hist_gp):
            st.session_state["rp_available_sessions"] = None
            st.session_state["rp_hist_analyser"] = None

        # Step 1 — Fetch available sessions for the selected GP
        if st.button("Fetch Sessions", key="rp_fetch_sessions"):
            with st.spinner("Fetching sessions from OpenF1..."):
                _client = OpenF1Client(mode="historical")
                _sessions = _client.get_sessions(hist_year, hist_gp)
            if _sessions.empty:
                st.error(
                    "No sessions found. Either the year/Grand Prix name is wrong, "
                    "or the OpenF1 API is unreachable (check your internet connection)."
                )
            else:
                st.session_state["rp_available_sessions"] = _sessions
                st.session_state["rp_fetched_for"] = (hist_year, hist_gp)

        # Step 2 — Session selector + Load button (shown once sessions are fetched)
        if st.session_state["rp_available_sessions"] is not None:
            _sessions_df = st.session_state["rp_available_sessions"]
            _session_names = _sessions_df["session_name"].tolist() if "session_name" in _sessions_df.columns else []
            # Default to Race if available
            _default_idx = next(
                (i for i, n in enumerate(_session_names) if "race" in str(n).lower() and "sprint" not in str(n).lower()),
                0,
            )
            hist_session_name = st.selectbox(
                "Session",
                options=_session_names,
                index=_default_idx,
                key="rp_session_name",
            )

            if st.button("Load Session Data", key="rp_load"):
                _row = _sessions_df[_sessions_df["session_name"] == hist_session_name].iloc[0]
                session_key = int(_row["session_key"])
                with st.spinner(f"Loading {hist_session_name} data from OpenF1..."):
                    client = OpenF1Client(mode="historical")
                    laps = client.get_laps(session_key)
                    stints = client.get_stints(session_key)
                    position = client.get_position(session_key)
                    drivers_df = client.get_drivers(session_key)

                if laps.empty:
                    st.error("No lap data returned for this session.")
                else:
                    driver_map: dict = {}
                    team_colour_map: dict = {}
                    if not drivers_df.empty and "driver_number" in drivers_df.columns and "name_acronym" in drivers_df.columns:
                        driver_map = {
                            int(row["driver_number"]): str(row["name_acronym"])
                            for _, row in drivers_df.iterrows()
                            if pd.notna(row["driver_number"]) and pd.notna(row["name_acronym"])
                        }
                        if "team_colour" in drivers_df.columns:
                            team_colour_map = {
                                int(row["driver_number"]): f"#{str(row['team_colour']).strip().upper()}"
                                for _, row in drivers_df.iterrows()
                                if pd.notna(row["driver_number"]) and pd.notna(row.get("team_colour"))
                            }
                    st.session_state["rp_hist_analyser"] = RaceAnalyser(laps, stints, position)
                    st.session_state["rp_hist_driver_map"] = driver_map
                    st.session_state["rp_hist_team_colour_map"] = team_colour_map
                    st.session_state["rp_hist_session_key"] = session_key
        else:
            st.info("Select a year and Grand Prix, then click **Fetch Sessions** to see available sessions.")

        # Render charts from session state if data has been loaded
        if st.session_state["rp_hist_analyser"] is not None:
            _analyser = st.session_state["rp_hist_analyser"]
            _dmap = st.session_state["rp_hist_driver_map"]
            _all_drvs = sorted(_analyser._clean["driver_number"].dropna().unique().tolist())
            # ---- Fastest Lap Telemetry Comparison ------------------
            st.markdown("### Fastest Lap Telemetry Comparison")
            _tc1, _tc2, _tc3 = st.columns(3)
            _tel_drv_a = _tc1.selectbox(
                "Driver A",
                options=_all_drvs,
                format_func=lambda n: _dmap.get(int(n), str(n)),
                key="rp_tel_drv_a",
            )
            _tel_drv_b = _tc2.selectbox(
                "Driver B",
                options=_all_drvs,
                index=min(1, len(_all_drvs) - 1),
                format_func=lambda n: _dmap.get(int(n), str(n)),
                key="rp_tel_drv_b",
            )
            _tel_channel = _tc3.selectbox(
                "Channel",
                options=SPECIAL_CHANNELS + list(TRACE_COLS.keys()),
                key="rp_tel_channel",
            )

            def _dlabel(n: int) -> str:
                return _dmap.get(n, str(n))

            if _tel_drv_a is None or _tel_drv_b is None:
                st.info("Select two drivers to compare telemetry.")
            else:
                _drv_a_int, _drv_b_int = int(_tel_drv_a), int(_tel_drv_b)
                _col_a, _col_b = _resolve_pair_colours(
                    _drv_a_int, _drv_b_int,
                    st.session_state.get("rp_hist_team_colour_map"),
                )
                _tel_cmap = _driver_color_map(
                    list(_all_drvs),
                    team_colour_map=st.session_state.get("rp_hist_team_colour_map"),
                )

                _hist_sk: int | str | None = st.session_state.get("rp_hist_session_key")
                _hist_laps = _analyser._laps

                if _hist_sk is None:
                    st.info("Session not loaded yet. Load a race first.")
                elif _tel_channel in SPECIAL_CHANNELS:
                    with st.spinner("Fetching fastest-lap telemetry from OpenF1..."):
                        _data_a = _fetch_fastest_lap_all_openf1(_hist_sk, _drv_a_int, _hist_laps)
                        _data_b = _fetch_fastest_lap_all_openf1(_hist_sk, _drv_b_int, _hist_laps)
                    if _data_a is None:
                        st.caption(f"Could not fetch telemetry for {_dlabel(_drv_a_int)} from OpenF1.")
                    elif _data_b is None:
                        st.caption(f"Could not fetch telemetry for {_dlabel(_drv_b_int)} from OpenF1.")
                    elif _tel_channel == "Track Map":
                        _gp_name = st.session_state.get("rp_gp", "")
                        _circuit = circuits.get(_gp_name)
                        if _circuit is None:
                            st.info(f"No circuit data for '{_gp_name}'. Run `python src/generate_circuits.py` to generate it.")
                        else:
                            _cx = np.asarray(_circuit["x"], dtype=float)
                            _cy = np.asarray(_circuit["y"], dtype=float)
                            _acronym_a, _acronym_b = _dlabel(_drv_a_int), _dlabel(_drv_b_int)
                            _map_a = {"x": _cx, "y": _cy, "speed": _data_a["speed"]}
                            _map_b = {"x": _cx, "y": _cy, "speed": _data_b["speed"]}
                            _special_fig = _build_track_map_fig(
                                _map_a, _acronym_a, _col_a,
                                _map_b, _acronym_b, _col_b,
                            )
                            if _special_fig is not None:
                                st.plotly_chart(_special_fig, width="stretch", key="tab3_track_map_hist")
                    elif _tel_channel == "Time Delta":
                        _special_fig = _build_time_delta_fig(
                            _data_a, _dlabel(_drv_a_int), _col_a,
                            _data_b, _dlabel(_drv_b_int), _col_b,
                        )
                        st.plotly_chart(_special_fig, width="stretch", key="tab3_time_delta_hist")
                else:
                    _tel_fig = go.Figure()
                    _any_trace = False
                    for _drv_int in (_drv_a_int, _drv_b_int):
                        with st.spinner(f"Fetching {_tel_channel} telemetry for {_dlabel(_drv_int)}..."):
                            _trace, _lap_t, _lap_num = _fetch_fastest_lap_openf1(
                                _hist_sk, _drv_int, _hist_laps, _tel_channel
                            )
                        if _trace is None:
                            st.caption(f"Could not fetch {_tel_channel} telemetry for {_dlabel(_drv_int)} from OpenF1.")
                            continue
                        _any_trace = True
                        _lap_label = f"Lap {_lap_num}, {_lap_t:.3f}s" if _lap_num is not None else f"{_lap_t:.3f}s"
                        _tel_fig.add_trace(go.Scatter(
                            x=np.arange(N_POINTS), y=_trace,
                            mode="lines",
                            name=f"{_dlabel(_drv_int)} ({_lap_label})",
                            line=dict(color=_tel_cmap.get(_drv_int, "#888"), width=2.5),
                        ))
                    if _any_trace:
                        _tel_fig.update_layout(
                            title=(
                                f"Fastest Lap {_tel_channel} — "
                                f"{_dlabel(_drv_a_int)} vs {_dlabel(_drv_b_int)}"
                            ),
                            xaxis_title="Normalised Distance (0–200 points)",
                            yaxis_title=_tel_channel,
                            legend_title="Driver",
                            hovermode="x unified",
                        )
                        st.plotly_chart(_tel_fig, width="stretch", key="tab3_tel_hist")

            st.divider()

            _selected = st.multiselect(
                "Drivers",
                options=_all_drvs,
                default=_all_drvs,
                format_func=lambda n: _dmap.get(int(n), str(n)),
                key="rp_driver_filter",
            )
            _render_charts(
                _analyser,
                0,
                st,
                driver_map=_dmap,
                selected_drivers=_selected or _all_drvs,
                team_colour_map=st.session_state.get("rp_hist_team_colour_map"),
            )

    else:
        # ---- LIVE MODE ----
        st.markdown(
            """
            <style>
            @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
            .live-badge {
                display: inline-block;
                font-size: 1.4rem;
                font-weight: 700;
                color: #ff4444;
                animation: pulse 1.5s ease-in-out infinite;
            }
            </style>
            <span class="live-badge">🔴 LIVE</span>
            """,
            unsafe_allow_html=True,
        )

        refresh_interval = st.selectbox(
            "Auto-refresh interval",
            options=[10, 30, 60],
            format_func=lambda s: f"{s}s",
            key="rp_refresh",
        )
        live_laps_remaining = st.number_input(
            "Estimated laps remaining", min_value=0, value=20, key="rp_live_laps",
        )

        chart_container = st.empty()

        try:
            # Persist live client across reruns so watermarks survive
            if st.session_state["rp_live_client"] is None:
                st.session_state["rp_live_client"] = OpenF1Client(mode="live")
            client = st.session_state["rp_live_client"]

            with st.spinner("Fetching live data from OpenF1..."):
                laps = client.get_live_laps()
                stints = client.get_live_stints()
                position = client.get_live_position()
                live_drivers_df = client.get_live_drivers()
                st.session_state["rp_last_refresh"] = datetime.now().strftime("%H:%M:%S")

            # Build / refresh driver map and team colour map
            if not live_drivers_df.empty and "driver_number" in live_drivers_df.columns and "name_acronym" in live_drivers_df.columns:
                st.session_state["rp_live_driver_map"] = {
                    int(row["driver_number"]): str(row["name_acronym"])
                    for _, row in live_drivers_df.iterrows()
                    if pd.notna(row["driver_number"]) and pd.notna(row["name_acronym"])
                }
                if "team_colour" in live_drivers_df.columns:
                    st.session_state["rp_live_team_colour_map"] = {
                        int(row["driver_number"]): f"#{str(row['team_colour']).strip().upper()}"
                        for _, row in live_drivers_df.iterrows()
                        if pd.notna(row["driver_number"]) and pd.notna(row.get("team_colour"))
                    }

            if laps.empty:
                st.warning("No active session found. Try during a race weekend.")
            else:
                analyser = RaceAnalyser(laps, stints, position)
                _live_dmap = st.session_state["rp_live_driver_map"]
                _all_drvs = sorted(analyser._clean["driver_number"].dropna().unique().tolist())

                # ---- Fastest Lap Telemetry Comparison (live) -------
                st.markdown("### Fastest Lap Telemetry Comparison")
                _lc1, _lc2, _lc3 = st.columns(3)
                _ltel_drv_a = _lc1.selectbox(
                    "Driver A",
                    options=_all_drvs,
                    format_func=lambda n: _live_dmap.get(int(n), str(n)),
                    key="rp_live_tel_drv_a",
                )
                _ltel_drv_b = _lc2.selectbox(
                    "Driver B",
                    options=_all_drvs,
                    index=min(1, len(_all_drvs) - 1),
                    format_func=lambda n: _live_dmap.get(int(n), str(n)),
                    key="rp_live_tel_drv_b",
                )
                _ltel_channel = _lc3.selectbox(
                    "Channel",
                    options=SPECIAL_CHANNELS + list(TRACE_COLS.keys()),
                    key="rp_live_tel_channel",
                )

                def _live_dlabel(n: int) -> str:
                    return _live_dmap.get(n, str(n))

                if _ltel_drv_a is None or _ltel_drv_b is None:
                    st.info("Select two drivers to compare telemetry.")
                else:
                    _ldrv_a_int, _ldrv_b_int = int(_ltel_drv_a), int(_ltel_drv_b)
                    _lcol_a, _lcol_b = _resolve_pair_colours(
                        _ldrv_a_int, _ldrv_b_int,
                        st.session_state.get("rp_live_team_colour_map"),
                    )
                    _ltel_cmap = _driver_color_map(
                        list(_all_drvs),
                        team_colour_map=st.session_state.get("rp_live_team_colour_map"),
                    )

                    # Live mode: use "latest" as the session key for car_data calls
                    _live_laps = analyser._laps

                    if _ltel_channel in SPECIAL_CHANNELS:
                        with st.spinner("Fetching fastest-lap telemetry from OpenF1..."):
                            _ldata_a = _fetch_fastest_lap_all_openf1("latest", _ldrv_a_int, _live_laps)
                            _ldata_b = _fetch_fastest_lap_all_openf1("latest", _ldrv_b_int, _live_laps)
                        if _ldata_a is None:
                            st.caption(f"Could not fetch telemetry for {_live_dlabel(_ldrv_a_int)} from OpenF1.")
                        elif _ldata_b is None:
                            st.caption(f"Could not fetch telemetry for {_live_dlabel(_ldrv_b_int)} from OpenF1.")
                        elif _ltel_channel == "Track Map":
                            _live_gp = st.session_state.get("rp_live_meeting_name", "")
                            _lcircuit = circuits.get(_live_gp) if _live_gp else None
                            if _lcircuit is None:
                                st.info("Circuit outline unavailable for the current live session.")
                            else:
                                _lcx = np.asarray(_lcircuit["x"], dtype=float)
                                _lcy = np.asarray(_lcircuit["y"], dtype=float)
                                _lacronym_a, _lacronym_b = _live_dlabel(_ldrv_a_int), _live_dlabel(_ldrv_b_int)
                                _lmap_a = {"x": _lcx, "y": _lcy, "speed": _ldata_a["speed"]}
                                _lmap_b = {"x": _lcx, "y": _lcy, "speed": _ldata_b["speed"]}
                                _lspecial_fig = _build_track_map_fig(
                                    _lmap_a, _lacronym_a, _lcol_a,
                                    _lmap_b, _lacronym_b, _lcol_b,
                                )
                                if _lspecial_fig is not None:
                                    st.plotly_chart(_lspecial_fig, width="stretch", key="tab3_track_map_live")
                        elif _ltel_channel == "Time Delta":
                            _lspecial_fig = _build_time_delta_fig(
                                _ldata_a, _live_dlabel(_ldrv_a_int), _lcol_a,
                                _ldata_b, _live_dlabel(_ldrv_b_int), _lcol_b,
                            )
                            st.plotly_chart(_lspecial_fig, width="stretch", key="tab3_time_delta_live")
                    else:
                        _ltel_fig = go.Figure()
                        _lany_trace = False
                        for _ldrv_int in (_ldrv_a_int, _ldrv_b_int):
                            with st.spinner(f"Fetching {_ltel_channel} telemetry for {_live_dlabel(_ldrv_int)}..."):
                                _ltrace, _llap_t, _llap_num = _fetch_fastest_lap_openf1(
                                    "latest", _ldrv_int, _live_laps, _ltel_channel
                                )
                            if _ltrace is None:
                                st.caption(f"Could not fetch {_ltel_channel} telemetry for {_live_dlabel(_ldrv_int)} from OpenF1.")
                                continue
                            _lany_trace = True
                            _llap_label = f"Lap {_llap_num}, {_llap_t:.3f}s" if _llap_num is not None else f"{_llap_t:.3f}s"
                            _ltel_fig.add_trace(go.Scatter(
                                x=np.arange(N_POINTS), y=_ltrace,
                                mode="lines",
                                name=f"{_live_dlabel(_ldrv_int)} ({_llap_label})",
                                line=dict(color=_ltel_cmap.get(_ldrv_int, "#888"), width=2.5),
                            ))
                        if _lany_trace:
                            _ltel_fig.update_layout(
                                title=(
                                    f"Fastest Lap {_ltel_channel} — "
                                    f"{_live_dlabel(_ldrv_a_int)} vs {_live_dlabel(_ldrv_b_int)}"
                                ),
                                xaxis_title="Normalised Distance (0–200 points)",
                                yaxis_title=_ltel_channel,
                                legend_title="Driver",
                                hovermode="x unified",
                            )
                            st.plotly_chart(_ltel_fig, width="stretch", key="tab3_tel_live")

                st.divider()

                _selected = st.multiselect(
                    "Drivers",
                    options=_all_drvs,
                    default=_all_drvs,
                    format_func=lambda n: _live_dmap.get(int(n), str(n)),
                    key="rp_driver_filter",
                )
                with chart_container.container():
                    _render_charts(
                        analyser,
                        live_laps_remaining,
                        st,
                        driver_map=_live_dmap,
                        selected_drivers=_selected or _all_drvs,
                        team_colour_map=st.session_state.get("rp_live_team_colour_map"),
                    )
        except Exception as exc:
            st.warning(f"Live refresh failed: {exc}")

        # Only auto-rerun if still on live mode — prevents resetting other tabs
        if st.session_state.get("rp_mode", "").startswith("📡"):
            time.sleep(refresh_interval)
            st.rerun()
