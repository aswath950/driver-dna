"""viz.py — Plotly figure builders and telemetry fetch helpers for the race dashboard.

No Streamlit UI code — all functions return go.Figure objects or plain data dicts
and are fully testable without a Streamlit runtime.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from openf1 import OpenF1Client
from features import N_POINTS

RACE_PALETTE = px.colors.qualitative.Dark24


def _build_throttle_map_fig(df: pd.DataFrame, selected: list[str], color_map: dict) -> go.Figure:
    """Build a multi-panel track map coloured by mean throttle per driver."""
    from plotly.subplots import make_subplots
    n = len(selected)
    fig = make_subplots(
        rows=1, cols=n,
        subplot_titles=selected,
        horizontal_spacing=0.04,
    )
    for i, drv in enumerate(selected, start=1):
        df_drv = df[df["driver"] == drv]
        try:
            x = np.nanmean(np.stack([np.array(r, dtype=float) for r in df_drv["x_trace"]]), axis=0)
            y = np.nanmean(np.stack([np.array(r, dtype=float) for r in df_drv["y_trace"]]), axis=0)
            thr = np.nanmean(np.stack([np.array(r, dtype=float) for r in df_drv["throttle_trace"]]), axis=0)
        except Exception:
            continue
        # Background outline
        fig.add_trace(
            go.Scatter(x=x, y=y, mode="lines",
                       line=dict(color="#333333", width=6),
                       showlegend=False, hoverinfo="skip"),
            row=1, col=i,
        )
        # Throttle-coloured layer
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="markers",
                marker=dict(
                    color=thr,
                    colorscale="RdYlGn",
                    cmin=0, cmax=100,
                    size=4,
                    colorbar=dict(title="Throttle %", thickness=12, len=0.6) if i == n else None,
                    showscale=(i == n),
                ),
                name=drv,
                showlegend=False,
                hovertemplate=f"<b>{drv}</b><br>Throttle: %{{marker.color:.0f}}%<extra></extra>",
            ),
            row=1, col=i,
        )
        axis_cfg = dict(visible=False, scaleanchor=f"y{i if i > 1 else ''}", scaleratio=1)
        fig.update_xaxes(axis_cfg, row=1, col=i)
        fig.update_yaxes(dict(visible=False), row=1, col=i)
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=30, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


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

    # Resample cumtime using raw sample index as the independent variable, not dist.
    # dist can be flat (non-increasing) in slow/stopped sections which makes it an
    # invalid xp for np.interp and produces incorrect interpolated cumtime values.
    # The sample index is always strictly increasing, so interpolation is well-defined.
    raw_idx = np.linspace(0.0, 1.0, len(cumtime_raw))
    grid_idx = np.linspace(0.0, 1.0, N_POINTS)

    return {
        "speed": _resample("speed"),
        "throttle": _resample("throttle"),
        "brake": _resample("brake"),
        "cumtime": np.interp(grid_idx, raw_idx, cumtime_raw),  # elapsed seconds at each distance point
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
        mode="markers",
        marker=dict(color="rgba(120,120,120,0.25)", size=8, symbol="circle"),
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
            mode="markers",
            marker=dict(color=col, size=7, symbol="circle"),
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
