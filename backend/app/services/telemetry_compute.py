"""Pure NumPy/Plotly algorithms for the fastest-lap telemetry charts.

Ported from ``src/viz.py`` — the legacy Streamlit implementation is the
canonical reference; this module reproduces the same post-fetch processing
(distance dedup, cumtime anchoring, microsector winner colouring, sector-
fraction overlays) so the industrialized backend produces byte-comparable
data and visually identical Plotly figures.

All functions are pure: no DB, no HTTP. The caller is responsible for
fetching the raw OpenF1 ``car_data`` DataFrame and the circuit geometry
(``x``, ``y``, ``sector_fractions``) from Postgres, then handing them in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Number of evenly-spaced distance samples per resampled trace. Raised from the
# legacy 200 (``src/features.N_POINTS`` / ``src/viz``) to 400 for finer corner
# detail — braking zones, apex minima and shift edges resolve at ~half the
# previous spacing. This is applied to the raw cached samples on every read, so
# the change needs no cache invalidation. ``corner_compute.N_POINTS`` MUST stay
# equal to this (apex fractions index the same grid); a test enforces it.
N_POINTS = 400

# Track-map upsample target (microsectors per circuit). Matches src/viz.
N_DISPLAY = 1000

# Thresholds for time-delta peak markers and "dead heat" labelling.
PEAK_DELTA_THRESHOLD_SEC = 0.05
DEAD_HEAT_THRESHOLD_SEC = 0.001

# OpenF1 car_data column names for each user-facing channel.
CHANNEL_COLUMN: dict[str, str] = {
    "Speed": "speed",
    "Throttle": "throttle",
    "Brake": "brake",
    "RPM": "rpm",
    "nGear": "n_gear",
    "DRS": "drs",
}

# Numeric car_data channels stored in the JSONB cache, in column order.
_SAMPLE_COLUMNS = ("speed", "throttle", "brake", "rpm", "n_gear", "drs")


# ---------------------------------------------------------------------------
# JSONB cache serialization
# ---------------------------------------------------------------------------


def _json_safe_column(values: object) -> list:
    """Coerce one car_data column to a JSON/JSONB-safe list of numbers/nulls.

    Real OpenF1 responses carry two kinds of value that break a JSONB write:
    ``pd.NA`` (missing columns are backfilled with it) isn't JSON-serializable
    at all, and ``NaN``/``±Inf`` (dropped or void samples) serialize to tokens
    Postgres JSONB rejects. Both are mapped to ``None`` (JSON ``null``), which
    JSONB accepts and the reconstruction path reads back as ``NaN`` — so the
    array stays index-aligned with ``dates``. Integral values are stored as
    ``int`` to keep the JSON compact (gear/DRS/rpm), others as ``float``.
    """
    nums = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    out: list = []
    for x in nums:
        if not np.isfinite(x):
            out.append(None)
        elif float(x).is_integer():
            out.append(int(x))
        else:
            out.append(float(x))
    return out


def samples_to_jsonb(car_df: pd.DataFrame) -> dict:
    """Serialize a raw car_data DataFrame to the columnar JSONB cache format.

    The single writer used by both the ETL bulk fetch and the request-time
    write-through cache, so every value that reaches ``car_telemetry.samples``
    is guaranteed JSON/JSONB-safe (see :func:`_json_safe_column`).
    """
    return {
        "dates": [d.isoformat() for d in car_df["date"]],
        **{col: _json_safe_column(car_df[col]) for col in _SAMPLE_COLUMNS},
    }


# ---------------------------------------------------------------------------
# Post-fetch processing — port of src/viz._fetch_fastest_lap_all_openf1
# ---------------------------------------------------------------------------


def process_car_data(
    car: pd.DataFrame,
    *,
    lap_duration: float,
    lap_end: pd.Timestamp | None = None,
    lap_number: int | None = None,
) -> dict | None:
    """Resample one driver's fastest-lap car_data to N_POINTS distance points.

    Algorithm — direct port of ``src/viz._fetch_fastest_lap_all_openf1`` from
    line 208 onwards:

    1. Sort by ``date``. If ``lap_end`` is given, clip rows beyond it (the
       OpenF1 query buffer can capture early next-lap samples).
    2. Build ``cumtime_raw`` from raw timestamp diffs, then rescale so
       ``cumtime_raw[-1] == lap_duration`` exactly (anchors to the official
       lap duration).
    3. Build a cumulative-distance proxy: ``dist = cumsum(speed_ms × dt)``.
    4. Resample speed/throttle/brake/rpm/n_gear/drs to N_POINTS evenly-spaced
       distance points via ``np.interp``.
    5. For cumtime, deduplicate ``dist`` with ``np.unique`` first — zero-speed
       sections produce non-increasing distance values that would otherwise
       cause spikes in the time delta.

    Returns ``None`` if the input has <2 usable samples.
    """
    if car.empty or "date" not in car.columns or "speed" not in car.columns:
        return None

    car = car.sort_values("date").reset_index(drop=True)

    if lap_end is not None:
        ts_end_clip = lap_end
        if car["date"].dt.tz is not None and ts_end_clip.tzinfo is None:
            ts_end_clip = ts_end_clip.tz_localize("UTC")
        car = car[car["date"] <= ts_end_clip].reset_index(drop=True)

    # Coerce so a pd.NA-backfilled speed column (OpenF1 omitted it) becomes NaN
    # instead of raising in .to_numpy(dtype=float); a no-op for real numeric data.
    speeds = pd.to_numeric(car["speed"], errors="coerce").to_numpy(dtype=float)
    if len(speeds) < 2:
        return None

    cumtime_raw = (
        (car["date"] - car["date"].iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    )
    if cumtime_raw[-1] > 0:
        cumtime_raw = cumtime_raw * (lap_duration / cumtime_raw[-1])

    dt = car["date"].diff().dt.total_seconds().fillna(0.0).to_numpy(dtype=float)[1:]
    speeds_ms = speeds[:-1] / 3.6
    dist_increments = np.where(
        np.isfinite(speeds_ms) & np.isfinite(dt), speeds_ms * dt, 0.0
    )
    dist = np.concatenate([[0.0], np.cumsum(dist_increments)])
    dist_grid = np.linspace(dist[0], dist[-1], N_POINTS)

    def _resample(col: str) -> np.ndarray:
        if col not in car.columns:
            return np.full(N_POINTS, np.nan)
        # ``pd.to_numeric(..., coerce)`` first: a column OpenF1 omitted is
        # backfilled with ``pd.NA`` by the client, and ``.to_numpy(dtype=float)``
        # raises on ``pd.NA``. Coercing maps NA/non-numeric to NaN, which
        # np.interp propagates as NaN samples rather than crashing the fetch.
        fp = pd.to_numeric(car[col], errors="coerce").to_numpy(dtype=float)
        return np.interp(dist_grid, dist, fp)

    # Dedup distance for cumtime interpolation (flat segments break np.interp).
    dist_unique, unique_idx = np.unique(dist, return_index=True)
    cumtime_unique = cumtime_raw[unique_idx]
    cumtime = np.interp(dist_grid, dist_unique, cumtime_unique)

    return {
        "speed": _resample("speed"),
        "throttle": _resample("throttle"),
        "brake": _resample("brake"),
        "rpm": _resample("rpm"),
        "n_gear": _resample("n_gear"),
        "drs": _resample("drs"),
        "cumtime": cumtime,
        "lap_time": float(lap_duration),
        "lap_number": lap_number,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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


def _lap_label(data: dict) -> str:
    """Format a driver's fastest lap as ``"Lap N, T.TTTs"`` (time only if no number)."""
    lap_n = data.get("lap_number")
    if lap_n is not None:
        return f"Lap {lap_n}, {data['lap_time']:.3f}s"
    return f"{data['lap_time']:.3f}s"


def corner_axis_ticks(
    corners: list[dict] | None,
    circuit_length_m: float | None,
    *,
    axis_max: float,
) -> tuple[list[float], list[str]] | None:
    """Map a circuit's corners to x-axis tick positions and turn labels.

    ``corners`` is the ``circuits.corners`` list — each entry has
    ``{"number", "letter", "distance_m"}`` where ``distance_m`` is metres from
    the start/finish line.  ``axis_max`` is the largest x value on the target
    axis (``N_POINTS - 1`` for the index-based channel charts, ``100`` for the
    lap-distance-% delta chart), so a corner at fraction ``f`` along the lap is
    placed at ``f * axis_max``.

    Returns ``(tickvals, ticktext)`` with one entry per in-bounds corner, or
    ``None`` when corner data is missing/unusable so callers can fall back to
    the plain normalised axis.
    """
    if not corners or not circuit_length_m or circuit_length_m <= 0:
        return None

    ordered = sorted(corners, key=lambda c: c["distance_m"])
    tickvals: list[float] = []
    ticktext: list[str] = []
    for c in ordered:
        frac = c["distance_m"] / circuit_length_m
        if not (0.0 <= frac <= 1.0):
            continue
        letter = str(c.get("letter", "")).strip()
        if letter.lower() in ("", "nan", "none"):
            letter = ""
        tickvals.append(round(frac * axis_max, 3))
        ticktext.append(f"T{int(c['number'])}{letter}")

    if not tickvals:
        return None
    return tickvals, ticktext


# ---------------------------------------------------------------------------
# Channel figure (Speed / Throttle / Brake / RPM / nGear / DRS)
# ---------------------------------------------------------------------------


def build_channel_figure(
    data_a: dict, acronym_a: str, color_a: str,
    data_b: dict, acronym_b: str, color_b: str,
    *,
    channel: str,
    corners: list[dict] | None = None,
    circuit_length_m: float | None = None,
) -> go.Figure:
    """Two-line per-channel comparison over lap distance.

    Mirrors the Streamlit chart at ``src/app.py:1990-2007`` — minimal,
    distance-indexed, hover-unified — but uses driver colours from the
    backend's team colour resolver.

    When ``corners`` + ``circuit_length_m`` are supplied the x-axis is
    annotated with the circuit's turn positions (T1, T2, …) instead of the raw
    ``0–N_POINTS`` normalised point indices; the chart falls back to the plain
    normalised axis when corner data is unavailable.  Speed is labelled in
    km/h since that is the unit of the underlying telemetry.
    """
    col = CHANNEL_COLUMN[channel]
    trace_a = data_a[col]
    trace_b = data_b[col]

    lap_a_num = data_a.get("lap_number")
    lap_b_num = data_b.get("lap_number")
    lap_a_str = (
        f"Lap {lap_a_num}, {data_a['lap_time']:.3f}s"
        if lap_a_num is not None
        else f"{data_a['lap_time']:.3f}s"
    )
    lap_b_str = (
        f"Lap {lap_b_num}, {data_b['lap_time']:.3f}s"
        if lap_b_num is not None
        else f"{data_b['lap_time']:.3f}s"
    )

    is_speed = channel == "Speed"
    y_title = "Speed (km/h)" if is_speed else channel
    # Speed values get an explicit km/h unit on hover; the driver name stays as
    # the per-row label (the <extra> slot) so the unified tooltip is unchanged
    # apart from the unit.
    hovertemplate = "%{y:.0f} km/h<extra>%{fullData.name}</extra>" if is_speed else None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(N_POINTS), y=trace_a,
        mode="lines",
        name=f"{acronym_a} ({lap_a_str})",
        line=dict(color=color_a, width=2.5),
        hovertemplate=hovertemplate,
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(N_POINTS), y=trace_b,
        mode="lines",
        name=f"{acronym_b} ({lap_b_str})",
        line=dict(color=color_b, width=2.5),
        hovertemplate=hovertemplate,
    ))

    ticks = corner_axis_ticks(corners, circuit_length_m, axis_max=N_POINTS - 1)
    if ticks is not None:
        tickvals, ticktext = ticks
        xaxis = dict(
            title="Track Position (turn)",
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=0,
        )
        # Subtle guide line at each turn so the apex positions read at a glance.
        for xv in tickvals:
            fig.add_vline(
                x=xv,
                line=dict(color="rgba(255,255,255,0.12)", width=1),
                layer="below",
            )
    else:
        xaxis = dict(title=f"Normalised Distance (0–{N_POINTS} points)")

    fig.update_layout(
        title=f"Fastest Lap {channel} — {acronym_a} vs {acronym_b}",
        xaxis=xaxis,
        yaxis_title=y_title,
        legend_title="Driver",
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(230, 230, 235, 0.97)",
            font=dict(color="rgba(15, 15, 20, 1)", size=12),
            bordercolor="rgba(190, 190, 200, 0.9)",
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------------
# Time-delta figure — port of src/viz._build_time_delta_fig
# ---------------------------------------------------------------------------


def build_time_delta_figure(
    data_a: dict, acronym_a: str, color_a: str,
    data_b: dict, acronym_b: str, color_b: str,
    *,
    sector_fractions: list[float] | None = None,
    corners: list[dict] | None = None,
    circuit_length_m: float | None = None,
) -> go.Figure:
    """Time delta over lap distance (0–100%) with sector-fraction overlays.

    When ``corners`` + ``circuit_length_m`` are supplied the x-axis ticks are
    relabelled to the circuit's turn positions (T1, T2, …) instead of the raw
    lap-distance percentages; the underlying x data stays in 0–100% so the
    hover read-outs are unaffected.
    """
    cumtime_a = data_a["cumtime"]
    cumtime_b = data_b["cumtime"]
    delta = cumtime_b - cumtime_a  # positive = A gaining

    n = len(delta)
    x_pct = np.linspace(0.0, 100.0, n)

    hover_text = [
        f"{acronym_a} ahead {abs(d):.3f}s" if d >= 0
        else f"{acronym_b} ahead {abs(d):.3f}s"
        for d in delta
    ]

    fig = go.Figure()

    pos = np.where(delta >= 0, delta, 0.0)
    neg = np.where(delta < 0, delta, 0.0)

    fig.add_trace(go.Scatter(
        x=np.concatenate([[x_pct[0]], x_pct, [x_pct[-1]]]),
        y=np.concatenate([[0], pos, [0]]),
        mode="lines",
        fill="tozeroy",
        fillcolor=_to_rgba(color_a, 0.30),
        line=dict(width=0),
        name=f"{acronym_a} ahead",
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
        name=f"{acronym_b} ahead",
        showlegend=False,
        hoverinfo="skip",
    ))

    delta_a = np.where(delta >= 0, delta, np.nan)
    delta_b = np.where(delta < 0, delta, np.nan)

    fig.add_trace(go.Scatter(
        x=x_pct, y=delta_a,
        mode="lines",
        connectgaps=False,
        line=dict(color=color_a, width=2.5),
        name=f"{acronym_a} ahead",
        customdata=hover_text,
        hovertemplate="%{x:.0f}%  —  %{customdata}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x_pct, y=delta_b,
        mode="lines",
        connectgaps=False,
        line=dict(color=color_b, width=2.5),
        name=f"{acronym_b} ahead",
        customdata=hover_text,
        hovertemplate="%{x:.0f}%  —  %{customdata}<extra></extra>",
    ))

    crossings = np.where(np.diff(np.sign(delta)) != 0)[0]
    if len(crossings) > 0:
        cx = x_pct[crossings]
        fig.add_trace(go.Scatter(
            x=cx, y=np.zeros(len(cx)),
            mode="markers",
            marker=dict(
                color="white", size=8, symbol="circle",
                line=dict(color="rgba(0,0,0,0.6)", width=1),
            ),
            name="Lead change",
            hovertemplate="Lead change at %{x:.0f}%<extra></extra>",
        ))

    max_delta = float(np.nanmax(delta))
    min_delta = float(np.nanmin(delta))

    if max_delta > PEAK_DELTA_THRESHOLD_SEC:
        idx_max = int(np.argmax(delta))
        fig.add_trace(go.Scatter(
            x=[x_pct[idx_max]], y=[max_delta],
            mode="markers",
            marker=dict(
                color=color_a, size=10, symbol="triangle-up",
                line=dict(color="white", width=1),
            ),
            name=f"Peak {acronym_a}",
            hovertemplate=(
                f"Peak {acronym_a} ahead {max_delta:.3f}s at %{{x:.0f}}%<extra></extra>"
            ),
        ))

    if min_delta < -PEAK_DELTA_THRESHOLD_SEC:
        idx_min = int(np.argmin(delta))
        fig.add_trace(go.Scatter(
            x=[x_pct[idx_min]], y=[min_delta],
            mode="markers",
            marker=dict(
                color=color_b, size=10, symbol="triangle-down",
                line=dict(color="white", width=1),
            ),
            name=f"Peak {acronym_b}",
            hovertemplate=(
                f"Peak {acronym_b} ahead {abs(min_delta):.3f}s at %{{x:.0f}}%<extra></extra>"
            ),
        ))

    final_gap = float(delta[-1])
    if abs(final_gap) < DEAD_HEAT_THRESHOLD_SEC:
        gap_text = "Dead heat"
        gap_color = "white"
    elif final_gap > 0:
        gap_text = f"{acronym_a} ahead\n{abs(final_gap):.3f}s"
        gap_color = color_a
    else:
        gap_text = f"{acronym_b} ahead\n{abs(final_gap):.3f}s"
        gap_color = color_b

    fig.add_annotation(
        x=x_pct[-1], y=final_gap,
        text=gap_text,
        showarrow=True,
        arrowhead=2,
        arrowcolor=gap_color,
        arrowwidth=1.5,
        ax=-55, ay=-35,
        font=dict(color=gap_color, size=12),
        xanchor="right",
        bgcolor="rgba(230, 230, 235, 0.92)",
        bordercolor=gap_color,
        borderwidth=1,
        borderpad=4,
    )

    if sector_fractions and len(sector_fractions) == 2:
        s1_frac, s2_frac = sector_fractions
        sector_starts = [
            (0.0, "S1"),
            (s1_frac * 100, "S2"),
            (s2_frac * 100, "S3"),
        ]
        for x_pos, label in sector_starts:
            fig.add_vline(
                x=x_pos,
                line=dict(color="rgba(255,255,255,0.45)", width=1.5, dash="dot"),
                layer="below",
            )
            fig.add_annotation(
                x=x_pos, y=1.0,
                xref="x", yref="paper",
                text=f"<b>{label}</b>",
                showarrow=False,
                xanchor="left",
                yanchor="top",
                font=dict(color="rgba(255,255,255,0.7)", size=10),
                xshift=4,
            )

    lap_a_num = data_a.get("lap_number")
    lap_b_num = data_b.get("lap_number")
    lap_a_str = (
        f"Lap {lap_a_num}, {data_a['lap_time']:.3f}s"
        if lap_a_num is not None
        else f"{data_a['lap_time']:.3f}s"
    )
    lap_b_str = (
        f"Lap {lap_b_num}, {data_b['lap_time']:.3f}s"
        if lap_b_num is not None
        else f"{data_b['lap_time']:.3f}s"
    )
    title_text = (
        f"Lap Time Delta — {acronym_a} vs {acronym_b}"
        f"<br><sup>{acronym_a}: {lap_a_str}  |  {acronym_b}: {lap_b_str}</sup>"
    )

    ticks = corner_axis_ticks(corners, circuit_length_m, axis_max=100.0)
    if ticks is not None:
        tickvals, ticktext = ticks
        xaxis = dict(
            title="Track Position (turn)", range=[0, 100], zeroline=False,
            tickmode="array", tickvals=tickvals, ticktext=ticktext, tickangle=0,
        )
    else:
        xaxis = dict(title="Lap Distance (%)", range=[0, 100], zeroline=False)

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14)),
        xaxis=xaxis,
        yaxis=dict(
            title=f"Gap (s)   ↑ {acronym_a} ahead   ·   {acronym_b} ahead ↓",
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.6)",
            zerolinewidth=1.5,
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(230, 230, 235, 0.97)",
            font=dict(color="rgba(15, 15, 20, 1)", size=12),
            bordercolor="rgba(190, 190, 200, 0.9)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=460,
        margin=dict(t=80, b=50, l=60, r=60),
    )
    return fig


# ---------------------------------------------------------------------------
# Speed + Time-delta combined figure — stacked subplots sharing the lap axis
# ---------------------------------------------------------------------------


def _momentum_bands(
    delta: np.ndarray,
    x_pct: np.ndarray,
    *,
    min_width_pct: float = 5.0,
    smooth_window: int = 11,
) -> list[tuple[float, float, str]]:
    """Segment the lap into "who is gaining time" bands for the momentum wash.

    The sign of the gap's slope says who is gaining through each stretch: a
    rising ``delta`` (= ``cumtime_b - cumtime_a``) means A is pulling away, a
    falling one means B is.  The raw per-sample sign flips far too often to read,
    so the gap is smoothed and same-sign runs are greedily merged until every
    band spans at least ``min_width_pct`` of the lap — the wash then reads as a
    handful of momentum zones rather than a barcode.

    Returns ``[(x0_pct, x1_pct, "a" | "b"), …]`` tiling ``[0, 100]`` with no
    gaps, where the label is the driver gaining time across that stretch.
    """
    n = len(delta)
    if n < 3:
        return []

    win = min(smooth_window, n if n % 2 == 1 else n - 1)
    smoothed = delta
    if win >= 3:
        smoothed = np.convolve(delta, np.ones(win) / win, mode="same")

    seg_gain = np.where(np.diff(smoothed) >= 0, "a", "b")  # one label per segment
    if len(seg_gain) == 0:
        return []

    def seg_width(start: int, end: int) -> float:
        return float(x_pct[end + 1] - x_pct[start])

    # Run-length encode contiguous same-gain segments into [start, end, label].
    runs: list[list] = []
    for i, gain in enumerate(seg_gain):
        if runs and runs[-1][2] == gain:
            runs[-1][1] = i
        else:
            runs.append([i, i, str(gain)])

    # Greedy left→right merge: absorb a thin run (or any run abutting a band
    # that is itself still too narrow) into the band being built, keeping that
    # band's hue, so every emitted band is wide enough to read.
    bands: list[list] = []  # [start_seg, end_seg, label]
    for start, end, label in runs:
        if bands:
            b_start, b_end, _ = bands[-1]
            if seg_width(b_start, b_end) < min_width_pct or seg_width(start, end) < min_width_pct:
                bands[-1][1] = end
                continue
        bands.append([start, end, label])

    return [(float(x_pct[s]), float(x_pct[e + 1]), lab) for s, e, lab in bands]


def build_speed_time_delta_figure(
    data_a: dict, acronym_a: str, color_a: str,
    data_b: dict, acronym_b: str, color_b: str,
    *,
    sector_fractions: list[float] | None = None,
    corners: list[dict] | None = None,
    circuit_length_m: float | None = None,
) -> go.Figure:
    """Stacked Speed (top) + cumulative Time-Delta (bottom) on a shared lap axis.

    Composes the two existing fastest-lap views into one figure so the *cause*
    (speed through a corner) sits directly above the *effect* (the gap it opens
    or closes).  Both panels share a single 0–100 % lap-distance x-axis — Speed
    is re-based from its native 0–``N_POINTS`` index — so features line up corner
    for corner, and a single vertical hover spike crosses both panels.

    A faint "momentum" wash on the speed panel tints each stretch by which driver
    is gaining time through it (see :func:`_momentum_bands`), so the speed trace
    visually explains the gap below it.  Turn ticks and sector boundaries are
    overlaid when circuit geometry is supplied; the chart degrades to a plain
    lap-distance axis without it.
    """
    speed_a = np.asarray(data_a["speed"], dtype=float)
    speed_b = np.asarray(data_b["speed"], dtype=float)
    delta = np.asarray(data_b["cumtime"], dtype=float) - np.asarray(
        data_a["cumtime"], dtype=float
    )  # positive = A ahead

    n = len(delta)
    x_pct = np.linspace(0.0, 100.0, n)
    lap_a_str, lap_b_str = _lap_label(data_a), _lap_label(data_b)

    # Both panels are placed on a SINGLE shared x-axis ("x") — speed on yaxis
    # "y" (top), gap on yaxis "y2" (bottom) — rather than via
    # ``make_subplots(shared_xaxes=True)``, which builds two *matched-but-
    # separate* x-axes.  Cross-panel hover (``hoversubplots="axis"``) only fires
    # when both panels reference the same x-axis, so a grid-"coupled" column with
    # every trace on ``xaxis="x"`` is what makes hovering one panel surface the
    # other's read-out at the same track position.
    top_domain, bot_domain = [0.46, 1.0], [0.0, 0.39]
    fig = go.Figure()

    # --- Speed traces (top panel, yaxis "y") ---
    speed_hover = "%{y:.0f} km/h<extra>%{fullData.name}</extra>"
    fig.add_trace(go.Scatter(
        x=x_pct, y=speed_a, mode="lines", xaxis="x", yaxis="y",
        name=f"{acronym_a} ({lap_a_str})", legendgroup="a",
        line=dict(color=color_a, width=2.5), hovertemplate=speed_hover,
    ))
    fig.add_trace(go.Scatter(
        x=x_pct, y=speed_b, mode="lines", xaxis="x", yaxis="y",
        name=f"{acronym_b} ({lap_b_str})", legendgroup="b",
        line=dict(color=color_b, width=2.5), hovertemplate=speed_hover,
    ))

    # --- Momentum wash on the speed panel (beneath the lines via layer="below") ---
    for x0, x1, gain in _momentum_bands(delta, x_pct):
        fig.add_shape(
            type="rect", xref="x", yref="y domain", x0=x0, x1=x1, y0=0, y1=1,
            fillcolor=_to_rgba(color_a if gain == "a" else color_b, 0.07),
            line_width=0, layer="below",
        )

    # --- Time-delta (bottom panel, yaxis "y2"): fills + signed lines + lead changes ---
    pos = np.where(delta >= 0, delta, 0.0)
    neg = np.where(delta < 0, delta, 0.0)
    x_fill = np.concatenate([[x_pct[0]], x_pct, [x_pct[-1]]])
    for fill_y, color in ((pos, color_a), (neg, color_b)):
        fig.add_trace(go.Scatter(
            x=x_fill, y=np.concatenate([[0], fill_y, [0]]), xaxis="x", yaxis="y2",
            mode="lines", fill="tozeroy", fillcolor=_to_rgba(color, 0.30),
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))

    hover_text = [
        f"{acronym_a} ahead {abs(d):.3f}s" if d >= 0
        else f"{acronym_b} ahead {abs(d):.3f}s"
        for d in delta
    ]
    for masked, color in (
        (np.where(delta >= 0, delta, np.nan), color_a),
        (np.where(delta < 0, delta, np.nan), color_b),
    ):
        fig.add_trace(go.Scatter(
            x=x_pct, y=masked, mode="lines", connectgaps=False, xaxis="x", yaxis="y2",
            line=dict(color=color, width=2.5), showlegend=False,
            customdata=hover_text,
            hovertemplate="%{customdata}<extra></extra>",
        ))

    crossings = np.where(np.diff(np.sign(delta)) != 0)[0]
    if len(crossings) > 0:
        cx = x_pct[crossings]
        fig.add_trace(go.Scatter(
            x=cx, y=np.zeros(len(cx)), mode="markers", name="Lead change",
            xaxis="x", yaxis="y2",
            marker=dict(
                color="white", size=8, symbol="circle",
                line=dict(color="rgba(0,0,0,0.6)", width=1),
            ),
            hovertemplate="Lead change at %{x:.0f}%<extra></extra>",
        ))

    # Final-gap call-out — who won the lap and by how much.
    final_gap = float(delta[-1])
    if abs(final_gap) < DEAD_HEAT_THRESHOLD_SEC:
        gap_text, gap_color = "Dead heat", "white"
    elif final_gap > 0:
        gap_text, gap_color = f"{acronym_a} +{abs(final_gap):.3f}s", color_a
    else:
        gap_text, gap_color = f"{acronym_b} +{abs(final_gap):.3f}s", color_b
    fig.add_annotation(
        xref="x", yref="y2", x=x_pct[-1], y=final_gap, text=gap_text,
        showarrow=True, arrowhead=2, arrowcolor=gap_color, arrowwidth=1.5,
        ax=-55, ay=-30, font=dict(color=gap_color, size=12), xanchor="right",
        bgcolor="rgba(230, 230, 235, 0.92)", bordercolor=gap_color,
        borderwidth=1, borderpad=4,
    )

    # --- Sector boundaries through both panels (when the circuit is seeded) ---
    if sector_fractions and len(sector_fractions) == 2:
        s1_frac, s2_frac = sector_fractions
        for x_pos, label in ((0.0, "S1"), (s1_frac * 100, "S2"), (s2_frac * 100, "S3")):
            for yref in ("y domain", "y2 domain"):
                fig.add_shape(
                    type="line", xref="x", yref=yref, x0=x_pos, x1=x_pos, y0=0, y1=1,
                    line=dict(color="rgba(255,255,255,0.45)", width=1.5, dash="dot"),
                    layer="below",
                )
            fig.add_annotation(
                xref="x", yref="y domain", x=x_pos, y=1.0, text=f"<b>{label}</b>",
                showarrow=False, xanchor="left", yanchor="top",
                font=dict(color="rgba(255,255,255,0.7)", size=10), xshift=4,
            )

    # --- Shared x-axis: turn ticks when corners are known, else lap-distance % ---
    ticks = corner_axis_ticks(corners, circuit_length_m, axis_max=100.0)
    xaxis = dict(
        domain=[0.0, 1.0], anchor="y2", range=[0, 100],
        title=dict(text="Track Position (turn)" if ticks else "Lap Distance (%)"),
        # Single hover spike spanning both stacked panels.
        showspikes=True, spikemode="across", spikesnap="cursor",
        spikethickness=1, spikecolor="rgba(255,255,255,0.35)", spikedash="solid",
    )
    if ticks is not None:
        tickvals, ticktext = ticks
        xaxis.update(tickmode="array", tickvals=tickvals, ticktext=ticktext, tickangle=0)

    fig.update_layout(
        title=dict(
            text=(
                f"Speed + Time Delta — {acronym_a} vs {acronym_b}"
                f"<br><sup>{acronym_a}: {lap_a_str}  |  {acronym_b}: {lap_b_str}"
                "   ·   speed shading = who's gaining time</sup>"
            ),
            font=dict(size=14),
        ),
        xaxis=xaxis,
        yaxis=dict(domain=top_domain, anchor="x", title=dict(text="Speed (km/h)")),
        yaxis2=dict(
            domain=bot_domain, anchor="x",
            title=dict(text=f"Gap (s)   ↑ {acronym_a}   ·   {acronym_b} ↓"),
            zeroline=True, zerolinecolor="rgba(255,255,255,0.6)", zerolinewidth=1.5,
        ),
        # Cross-panel hover: ``hoversubplots="axis"`` pulls every trace on the
        # shared x-axis into one unified tooltip; the grid-"coupled" column is
        # what marks the two panels as a single stack for that to take effect.
        grid=dict(rows=2, columns=1, pattern="coupled"),
        hovermode="x unified",
        hoversubplots="axis",
        hoverlabel=dict(
            bgcolor="rgba(230, 230, 235, 0.97)",
            font=dict(color="rgba(15, 15, 20, 1)", size=12),
            bordercolor="rgba(190, 190, 200, 0.9)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=560,
        margin=dict(t=90, b=50, l=70, r=60),
    )
    return fig


# ---------------------------------------------------------------------------
# Sector times — port of src/viz._build_sector_times_fig
# ---------------------------------------------------------------------------


def _sector_fallback_figure(msg: str) -> go.Figure:
    f = go.Figure()
    f.add_annotation(
        text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(color="white", size=14),
    )
    f.update_layout(
        height=460, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, b=40, l=60, r=40),
    )
    return f


def compute_sector_times(
    cumtime: np.ndarray, sector_fractions: list[float]
) -> tuple[float, float, float] | None:
    """Return (s1, s2, s3) seconds from a cumtime array and sector fractions.

    None if any sector is non-positive or non-finite.
    """
    if not sector_fractions or len(sector_fractions) != 2:
        return None
    n = len(cumtime)
    s1_frac, s2_frac = sector_fractions
    i1 = int(round(s1_frac * (n - 1)))
    i2 = int(round(s2_frac * (n - 1)))
    s1 = float(cumtime[i1])
    s2 = float(cumtime[i2] - cumtime[i1])
    s3 = float(cumtime[-1] - cumtime[i2])
    if any(not np.isfinite(v) or v <= 0 for v in (s1, s2, s3)):
        return None
    return s1, s2, s3


def build_sector_times_figure(
    data_a: dict, acronym_a: str, color_a: str,
    data_b: dict, acronym_b: str, color_b: str,
    *,
    sector_fractions: list[float] | None = None,
) -> tuple[go.Figure, tuple[float, float, float] | None, tuple[float, float, float] | None]:
    """Grouped bar chart of S1/S2/S3 times with delta annotations.

    Returns ``(figure, splits_a, splits_b)`` where each splits tuple is
    ``(s1_sec, s2_sec, s3_sec)`` or ``None`` if the data is unusable.
    """
    if not sector_fractions or len(sector_fractions) != 2:
        return (
            _sector_fallback_figure(
                "Sector boundaries not available — circuit not seeded"
            ),
            None,
            None,
        )

    splits_a = compute_sector_times(data_a["cumtime"], sector_fractions)
    splits_b = compute_sector_times(data_b["cumtime"], sector_fractions)
    if splits_a is None or splits_b is None:
        return (
            _sector_fallback_figure(
                "Sector time data is inconsistent — telemetry may be incomplete"
            ),
            splits_a,
            splits_b,
        )

    s1a, s2a, s3a = splits_a
    s1b, s2b, s3b = splits_b
    times_a = [s1a, s2a, s3a]
    times_b = [s1b, s2b, s3b]
    sectors = ["S1", "S2", "S3"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sectors, y=times_a, name=acronym_a, marker_color=color_a,
        text=[f"{t:.3f}s" for t in times_a], textposition="outside",
        hovertemplate="%{x}: %{y:.3f}s<extra>" + acronym_a + "</extra>",
    ))
    fig.add_trace(go.Bar(
        x=sectors, y=times_b, name=acronym_b, marker_color=color_b,
        text=[f"{t:.3f}s" for t in times_b], textposition="outside",
        hovertemplate="%{x}: %{y:.3f}s<extra>" + acronym_b + "</extra>",
    ))

    for sector_name, ta, tb in zip(sectors, times_a, times_b):
        d = tb - ta  # positive = A faster
        if abs(d) < DEAD_HEAT_THRESHOLD_SEC:
            ann_text, ann_color = "Dead heat", "white"
        elif d > 0:
            ann_text, ann_color = f"{acronym_a} ahead {d:.3f}s", color_a
        else:
            ann_text, ann_color = f"{acronym_b} ahead {abs(d):.3f}s", color_b
        fig.add_annotation(
            x=sector_name, y=max(ta, tb), text=ann_text,
            showarrow=False, yshift=28,
            font=dict(color=ann_color, size=11), xanchor="center",
        )

    total_delta = data_b["lap_time"] - data_a["lap_time"]
    if abs(total_delta) < DEAD_HEAT_THRESHOLD_SEC:
        total_text, total_color = "Dead heat overall", "white"
    elif total_delta > 0:
        total_text, total_color = (
            f"{acronym_a} ahead {total_delta:.3f}s overall",
            color_a,
        )
    else:
        total_text, total_color = (
            f"{acronym_b} ahead {abs(total_delta):.3f}s overall",
            color_b,
        )
    fig.add_annotation(
        text=f"<b>{total_text}</b>",
        xref="paper", yref="paper", x=0.5, y=-0.18,
        xanchor="center", yanchor="top",
        showarrow=False,
        font=dict(color=total_color, size=13),
    )

    lap_a_num = data_a.get("lap_number")
    lap_b_num = data_b.get("lap_number")
    lap_a_str = (
        f"Lap {lap_a_num}, {data_a['lap_time']:.3f}s"
        if lap_a_num is not None
        else f"{data_a['lap_time']:.3f}s"
    )
    lap_b_str = (
        f"Lap {lap_b_num}, {data_b['lap_time']:.3f}s"
        if lap_b_num is not None
        else f"{data_b['lap_time']:.3f}s"
    )

    fig.update_layout(
        title=dict(
            text=(
                f"Sector Times — {acronym_a} vs {acronym_b}"
                f"<br><sup>{acronym_a}: {lap_a_str}  |  {acronym_b}: {lap_b_str}</sup>"
            ),
            font=dict(size=14),
        ),
        barmode="group",
        xaxis=dict(title="Sector"),
        yaxis=dict(title="Time (s)", rangemode="tozero"),
        hoverlabel=dict(
            bgcolor="rgba(230, 230, 235, 0.97)",
            font=dict(color="rgba(15, 15, 20, 1)", size=12),
            bordercolor="rgba(190, 190, 200, 0.9)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=460,
        margin=dict(t=80, b=75, l=60, r=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig, splits_a, splits_b


# ---------------------------------------------------------------------------
# Track map — port of src/viz._build_track_map_fig
# ---------------------------------------------------------------------------


def _rotate_circuit_to_fit(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    target_aspect: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a circuit so its principal axis fills a landscape container.

    Tries all four 90-degree orientations of the PCA principal axis and
    picks the one whose bounding-box aspect ratio is closest to
    ``target_aspect`` (width / height).
    """
    cx, cy = x_arr.mean(), y_arr.mean()
    xc = x_arr - cx
    yc = y_arr - cy

    cov = np.cov(xc, yc)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal = eigenvectors[:, np.argmax(eigenvalues)]
    base_angle = np.arctan2(principal[1], principal[0])

    best_angle = 0.0
    best_waste = float("inf")
    for k in range(4):
        angle = -(base_angle + k * np.pi / 2)
        cos_t, sin_t = np.cos(angle), np.sin(angle)
        xr = cos_t * xc - sin_t * yc
        yr = sin_t * xc + cos_t * yc
        w = xr.max() - xr.min()
        h = yr.max() - yr.min()
        if h == 0:
            continue
        aspect = w / h
        waste = abs(np.log(aspect / target_aspect))
        if waste < best_waste:
            best_waste = waste
            best_angle = angle

    cos_t, sin_t = np.cos(best_angle), np.sin(best_angle)
    return cos_t * xc - sin_t * yc, sin_t * xc + cos_t * yc


def build_track_map_figure(
    data_a: dict, acronym_a: str, color_a: str,
    data_b: dict, acronym_b: str, color_b: str,
    *,
    circuit_x: list[float] | None,
    circuit_y: list[float] | None,
    sector_fractions: list[float] | None = None,
) -> go.Figure | None:
    """Track outline coloured by the faster driver per microsector.

    Returns ``None`` if circuit geometry is missing.
    """
    if not circuit_x or not circuit_y:
        return None

    x_arr = np.asarray(circuit_x, dtype=float)
    y_arr = np.asarray(circuit_y, dtype=float)

    # Rotate circuit to maximise use of the landscape canvas.
    x_arr, y_arr = _rotate_circuit_to_fit(x_arr, y_arr)

    spd_a = np.asarray(data_a["speed"], dtype=float)
    spd_b = np.asarray(data_b["speed"], dtype=float)

    t_fine = np.linspace(0.0, 1.0, N_DISPLAY)

    t_xy = np.linspace(0.0, 1.0, len(x_arr))
    x_fine = np.interp(t_fine, t_xy, x_arr)
    y_fine = np.interp(t_fine, t_xy, y_arr)

    t_spd = np.linspace(0.0, 1.0, len(spd_a))
    spd_a_fine = np.interp(t_fine, t_spd, spd_a)
    spd_b_fine = np.interp(t_fine, t_spd, spd_b)

    winner = np.where(spd_a_fine[:-1] >= spd_b_fine[:-1], "a", "b")

    # Tight axis bounds — remove Plotly's default padding so the circuit
    # fills the canvas after rotation.
    pad_x = (x_fine.max() - x_fine.min()) * 0.05
    pad_y = (y_fine.max() - y_fine.min()) * 0.05
    x_range = [float(x_fine.min() - pad_x), float(x_fine.max() + pad_x)]
    y_range = [float(y_fine.min() - pad_y), float(y_fine.max() + pad_y)]

    fig = go.Figure()

    # --- Circle + grey line style ---
    # Layer 1: one circle per microsector coloured by the faster driver.
    # A single trace with a per-point colour array is used so there are no
    # run-length segments — every point renders independently and gap-free.
    colors_per_point = [color_a if w == "a" else color_b for w in winner]

    # Invisible legend proxy traces so the driver names appear in the legend.
    for key, col, label in [("a", color_a, acronym_a), ("b", color_b, acronym_b)]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(color=col, size=10, symbol="circle"),
            name=label,
            legendgroup=key,
        ))

    fig.add_trace(go.Scatter(
        x=x_fine[:-1], y=y_fine[:-1],
        mode="markers",
        marker=dict(color=colors_per_point, size=12, symbol="circle"),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Layer 2: grey line drawn on top — runs through the centre of every
    # circle, leaving the coloured ring visible around each side.
    fig.add_trace(go.Scatter(
        x=x_fine, y=y_fine,
        mode="lines",
        line=dict(color="rgba(110,110,120,0.90)", width=5),
        showlegend=False,
        hoverinfo="skip",
    ))

    if sector_fractions and len(sector_fractions) == 2:
        s1_frac, s2_frac = sector_fractions
        boundaries = [(0.0, "S1"), (s1_frac, "S2"), (s2_frac, "S3")]
        for frac, label in boundaries:
            idx = int(round(frac * (N_DISPLAY - 1)))
            bx, by = float(x_fine[idx]), float(y_fine[idx])
            fig.add_trace(go.Scatter(
                x=[bx], y=[by],
                mode="markers+text",
                marker=dict(
                    color="white", size=14, symbol="circle",
                    line=dict(color="rgba(0,0,0,0.8)", width=2),
                ),
                text=[f"<b>{label}</b>"],
                textposition="top center",
                textfont=dict(color="white", size=11),
                name=label,
                showlegend=True,
                legendgroup="sectors",
                legendgrouptitle_text="Sectors" if label == "S1" else None,
                hovertemplate=f"{label} boundary<extra></extra>",
            ))

    fig.update_layout(
        title=(
            f"Track Map — {acronym_a} vs {acronym_b} "
            f"(faster driver by microsector)"
        ),
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1, range=x_range),
        yaxis=dict(visible=False, range=y_range),
        legend_title="Faster driver",
        hoverlabel=dict(
            bgcolor="rgba(230, 230, 235, 0.97)",
            font=dict(color="rgba(15, 15, 20, 1)", size=12),
            bordercolor="rgba(190, 190, 200, 0.9)",
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=40, b=0),
        height=520,
    )
    return fig
