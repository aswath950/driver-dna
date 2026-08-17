"""Pure computation for corner detection, metrics, and Plotly figures.

No DB access, no HTTP calls. All inputs are plain Python / NumPy arrays so
every function is unit-testable in isolation.

Corner detection uses the circuit's x/y outline (from the circuits table) to
compute curvature, find apex peaks, and derive entry/exit boundaries.  The
resulting corner map is cached per-circuit via lru_cache since circuit geometry
is immutable within a process lifetime.

Corner positions are expressed as 0–1 normalized arc-length fractions that
directly index into the N_POINTS=200 distance grid produced by
telemetry_compute.process_car_data().
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# Must match telemetry_compute.N_POINTS so apex fractions index the same grid.
# Kept as a literal (not imported) to keep this module free of the plotly-heavy
# telemetry_compute import; a test asserts the two stay equal.
N_POINTS = 400

CornerClass = Literal["slow", "medium", "high"]


# ---------------------------------------------------------------------------
# Corner detection — cached per circuit geometry
# ---------------------------------------------------------------------------


@lru_cache(maxsize=50)
def detect_corners(
    x: tuple[float, ...],
    y: tuple[float, ...],
) -> list[dict]:
    """Detect corners from circuit outline coordinates.

    Args:
        x, y: Circuit outline point arrays as tuples (hashable for lru_cache).

    Returns:
        List of dicts with keys: corner_number, apex_frac, entry_frac, exit_frac.
        All *_frac values are 0–1 arc-length fractions.
    """
    xa, ya = np.array(x, dtype=float), np.array(y, dtype=float)

    # Arc-length from start
    ds = np.sqrt(np.diff(xa) ** 2 + np.diff(ya) ** 2)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    total = s[-1]
    if total < 1e-6:
        return []
    s_norm = s / total  # 0–1 fraction

    # Curvature: κ = |x'y'' - y'x''| / (x'²+y'²)^1.5
    dx = np.gradient(xa)
    dy = np.gradient(ya)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx ** 2 + dy ** 2) ** 1.5
    denom = np.where(denom < 1e-8, 1e-8, denom)
    kappa = np.abs(dx * ddy - dy * ddx) / denom

    # Smooth to suppress noise; sigma≈1% of array length
    sigma = max(3, len(kappa) // 80)
    kappa_smooth = gaussian_filter1d(kappa, sigma=sigma)

    # Peak detection: height ≥ 75th-percentile, min distance ≈ 2% of track
    min_height = float(np.percentile(kappa_smooth, 75))
    min_distance = max(10, len(kappa_smooth) // 50)
    peaks, _ = find_peaks(kappa_smooth, height=min_height, distance=min_distance)

    corners: list[dict] = []
    for i, peak_idx in enumerate(peaks):
        peak_k = kappa_smooth[peak_idx]
        threshold = 0.3 * peak_k

        # Walk backward to entry
        entry_idx = int(peak_idx)
        while entry_idx > 0 and kappa_smooth[entry_idx] > threshold:
            entry_idx -= 1

        # Walk forward to exit
        exit_idx = int(peak_idx)
        while exit_idx < len(kappa_smooth) - 1 and kappa_smooth[exit_idx] > threshold:
            exit_idx += 1

        corners.append({
            "corner_number": i + 1,
            "apex_frac":  float(s_norm[peak_idx]),
            "entry_frac": float(s_norm[entry_idx]),
            "exit_frac":  float(s_norm[min(exit_idx, len(s_norm) - 1)]),
        })

    return corners


# ---------------------------------------------------------------------------
# Bounded entry/exit boundaries — shared by both detection paths
# ---------------------------------------------------------------------------


def _territory_bounds(apex_indices: list[int], n: int) -> list[tuple[int, int]]:
    """Split the speed grid into disjoint per-corner territories.

    Each corner owns the span up to the midpoint between its apex and each
    neighbour, so no two corners' entry/exit windows can overlap (the cause of
    duplicate v_min values when corners are close together — chicanes, esses).

    Args:
        apex_indices: Apex grid indices, ascending. May contain equal/adjacent
                      values on long circuits where corners round to the same
                      200-pt index.
        n:            Length of the speed grid (N_POINTS).

    Returns:
        List of (lower, upper) bounds aligned 1-to-1 with apex_indices, each
        guaranteed to satisfy ``lower <= apex <= upper`` (window size >= 1).
    """
    k = len(apex_indices)
    bounds: list[tuple[int, int]] = []
    for i, apex in enumerate(apex_indices):
        lower = 0 if i == 0 else (apex_indices[i - 1] + apex) // 2 + 1
        upper = n - 1 if i == k - 1 else (apex + apex_indices[i + 1]) // 2
        # Guard coincident/adjacent apexes so the apex always stays inside its
        # own territory and the window never collapses below one point.
        lower = min(lower, apex)
        upper = max(upper, apex)
        bounds.append((lower, upper))
    return bounds


def _corner_boundaries(
    sp_smooth: np.ndarray,
    apex_idx: int,
    lower: int,
    upper: int,
) -> tuple[int, int]:
    """Find entry/exit indices around an apex, clamped to a territory.

    Walks outward from the apex while speed stays within 30% recovery of the
    surrounding approach/exit rise, but never past ``lower`` (entry) or
    ``upper`` (exit) — both the walk and the rise reference are confined to the
    corner's own territory.

    Returns:
        (entry_idx, exit_idx) with ``lower <= entry_idx <= apex_idx <= exit_idx <= upper``.
    """
    v_apex = sp_smooth[apex_idx]

    # Entry: walk backward while still within 30% recovery of approach drop.
    # The 50-pt rise reference is clamped to the territory (lower instead of 0)
    # so it never reaches across into a neighbouring corner.
    approach_max = float(sp_smooth[max(lower, apex_idx - 50) : apex_idx + 1].max())
    entry_threshold = v_apex + 0.3 * (approach_max - v_apex)
    entry_idx = int(apex_idx)
    while entry_idx > lower and sp_smooth[entry_idx] < entry_threshold:
        entry_idx -= 1

    # Exit: walk forward while still within 30% recovery of exit rise.
    exit_max = float(sp_smooth[apex_idx : min(upper + 1, apex_idx + 50)].max())
    exit_threshold = v_apex + 0.3 * (exit_max - v_apex)
    exit_idx = int(apex_idx)
    while exit_idx < upper and sp_smooth[exit_idx] < exit_threshold:
        exit_idx += 1

    return entry_idx, exit_idx


# ---------------------------------------------------------------------------
# Preloaded-corner mapping (FastF1 authoritative data)
# ---------------------------------------------------------------------------


def corners_from_preloaded(
    preloaded: list[dict],
    circuit_length_m: float,
    speed: list[float],
) -> list[dict]:
    """Convert FastF1 corner records to corner dicts ready for classify_corners.

    Each element of ``preloaded`` is expected to have keys::

        {"number": int, "letter": str, "distance_m": float}

    The ``distance_m`` values are metres from the start/finish line (as stored
    by FastF1) and are divided by ``circuit_length_m`` to produce normalised
    0–1 arc-length fractions that index into the N_POINTS=200 telemetry grid.
    Entry and exit boundaries are still derived from the speed profile using
    the same 30%-recovery walk as detect_corners_from_speed.

    Args:
        preloaded:        List of corner dicts from ``circuits.corners``.
        circuit_length_m: Total circuit length in metres (``length_km * 1000``).
        speed:            N_POINTS=200 speed array (km/h) from process_car_data().

    Returns:
        List of dicts: {corner_number, apex_frac, entry_frac, exit_frac}.
    """
    sp = np.array(speed, dtype=float)
    n = len(sp)
    sp_smooth = gaussian_filter1d(sp, sigma=3)

    ordered = sorted(preloaded, key=lambda r: r["distance_m"])
    apex_fracs = [
        float(np.clip(c["distance_m"] / circuit_length_m, 0.0, 1.0)) for c in ordered
    ]
    apex_indices = [int(np.clip(round(f * (n - 1)), 0, n - 1)) for f in apex_fracs]
    # Disjoint per-corner territories keep each entry/exit walk off its
    # neighbours, so close corners (chicanes) no longer share a v_min.
    bounds = _territory_bounds(apex_indices, n)

    corners: list[dict] = []
    for c, apex_frac, apex_idx, (lower, upper) in zip(
        ordered, apex_fracs, apex_indices, bounds
    ):
        entry_idx, exit_idx = _corner_boundaries(sp_smooth, apex_idx, lower, upper)
        corners.append({
            "corner_number": int(c["number"]),
            "apex_frac":  apex_frac,
            "entry_frac": float(entry_idx / (n - 1)),
            "exit_frac":  float(exit_idx / (n - 1)),
        })

    return corners


# ---------------------------------------------------------------------------
# Telemetry-based corner detection
# ---------------------------------------------------------------------------


def detect_corners_from_speed(speed: list[float]) -> list[dict]:
    """Detect corners as local speed minima in the telemetry speed array.

    All *_frac values are normalised indices into the N_POINTS speed grid
    (apex_frac = apex_idx / (N - 1)), the same coordinate space used by
    classify_corners and compute_corner_metrics.  This avoids the
    circuit-geometry arc-length ↔ telemetry-distance misalignment that
    caused wrong classifications when the GPS outline starts at a different
    point on the circuit than the lap start/finish line.

    Args:
        speed: N_POINTS=200 speed array (km/h) from process_car_data().

    Returns:
        List of dicts: {corner_number, apex_frac, entry_frac, exit_frac}.
    """
    sp = np.array(speed, dtype=float)
    n = len(sp)

    sp_smooth = gaussian_filter1d(sp, sigma=3)

    # prominence >= 15 km/h filters out noise and minor kinks;
    # distance >= 8 points (~4% of a lap) prevents double-detection of one corner.
    peaks, _ = find_peaks(-sp_smooth, prominence=15, distance=8)

    apex_indices = [int(p) for p in peaks]
    # Disjoint per-corner territories keep each entry/exit walk off its
    # neighbours, so close corners (chicanes) no longer share a v_min.
    bounds = _territory_bounds(apex_indices, n)

    corners: list[dict] = []
    for i, (apex_idx, (lower, upper)) in enumerate(zip(apex_indices, bounds)):
        entry_idx, exit_idx = _corner_boundaries(sp_smooth, apex_idx, lower, upper)
        corners.append({
            "corner_number": i + 1,
            "apex_frac":  float(apex_idx / (n - 1)),
            "entry_frac": float(entry_idx / (n - 1)),
            "exit_frac":  float(exit_idx / (n - 1)),
        })

    return corners


# ---------------------------------------------------------------------------
# Corner classification
# ---------------------------------------------------------------------------


def classify_corners(
    corners: list[dict],
    ref_speed: list[float],
) -> list[dict]:
    """Add corner_class and ref_speed_kmh to each corner dict.

    Args:
        corners:   Output of detect_corners().
        ref_speed: N_POINTS=200 speed array (km/h) from process_car_data().
                   Used as reference to determine apex speed.

    Returns:
        New list of dicts with added keys: corner_class, ref_speed_kmh.
    """
    speed_arr = np.array(ref_speed, dtype=float)
    n = len(speed_arr)
    result: list[dict] = []
    for c in corners:
        apex_idx = int(np.clip(round(c["apex_frac"] * (n - 1)), 0, n - 1))
        v = float(speed_arr[apex_idx])
        cls: CornerClass = "slow" if v < 100 else ("medium" if v < 180 else "high")
        result.append({**c, "corner_class": cls, "ref_speed_kmh": round(v, 1)})
    return result


# ---------------------------------------------------------------------------
# Per-driver corner metrics
# ---------------------------------------------------------------------------


def compute_corner_metrics(
    speed: list[float],
    throttle: list[float],
    brake: list[float],
    corners: list[dict],
) -> list[dict]:
    """Extract performance metrics for one driver at each corner.

    Args:
        speed, throttle, brake: N_POINTS=200 arrays from process_car_data().
        corners: Classified corner list (output of classify_corners()).

    Returns:
        List of dicts aligned 1-to-1 with corners:
            corner_number, v_min (km/h), exit_speed (km/h),
            throttle_dist_frac (0–1), brake_point_frac (0–1).
    """
    sp = np.array(speed, dtype=float)
    th = np.array(throttle, dtype=float)
    br = np.array(brake, dtype=float)
    n = len(sp)

    metrics: list[dict] = []
    for c in corners:
        entry_i = int(np.clip(round(c["entry_frac"] * (n - 1)), 0, n - 1))
        apex_i  = int(np.clip(round(c["apex_frac"]  * (n - 1)), 0, n - 1))
        exit_i  = int(np.clip(round(c["exit_frac"]  * (n - 1)), 0, n - 1))

        # Ensure valid ordering (floating-point rounding can make them equal)
        entry_i = min(entry_i, apex_i)
        exit_i  = max(apex_i, exit_i)
        corner_len = max(exit_i - entry_i, 1)

        # v_min: minimum speed in [entry, exit]
        v_min = float(sp[entry_i : exit_i + 1].min())

        # exit_speed: speed at midpoint between apex and exit
        mid_exit_i = int(np.clip(apex_i + (exit_i - apex_i) // 2, 0, n - 1))
        exit_speed = float(sp[mid_exit_i])

        # throttle_dist_frac: first throttle >10% after apex, as fraction of corner
        thr_slice = th[apex_i : exit_i + 1]
        thr_open = np.where(thr_slice > 10)[0]
        thr_frac = float(thr_open[0] / corner_len) if len(thr_open) else 1.0

        # brake_point_frac: distance of last brake >5% before apex, as fraction
        brk_slice = br[entry_i : apex_i + 1]
        brk_pts = np.where(brk_slice > 5)[0]
        brk_frac = (
            float((apex_i - (entry_i + int(brk_pts[-1]))) / corner_len)
            if len(brk_pts) else 0.0
        )

        metrics.append({
            "corner_number":      c["corner_number"],
            "v_min":              round(v_min, 2),
            "exit_speed":         round(exit_speed, 2),
            "throttle_dist_frac": round(float(np.clip(thr_frac, 0.0, 1.0)), 4),
            "brake_point_frac":   round(float(np.clip(brk_frac, 0.0, 1.0)), 4),
        })
    return metrics


# ---------------------------------------------------------------------------
# Team aggregation
# ---------------------------------------------------------------------------


def aggregate_team_metrics(
    driver_metrics_list: list[list[dict]],
) -> list[dict]:
    """Median-aggregate corner metrics across 1 or 2 drivers.

    Args:
        driver_metrics_list: 1 or 2 per-driver metric lists from
                             compute_corner_metrics().  All lists must have
                             the same corner count and ordering.

    Returns:
        Single aggregated metric list (same shape as input lists).
    """
    if len(driver_metrics_list) == 1:
        return driver_metrics_list[0]

    keys = ["v_min", "exit_speed", "throttle_dist_frac", "brake_point_frac"]
    n_corners = len(driver_metrics_list[0])
    result: list[dict] = []
    for ci in range(n_corners):
        row: dict = {"corner_number": driver_metrics_list[0][ci]["corner_number"]}
        for k in keys:
            vals = [dm[ci][k] for dm in driver_metrics_list if ci < len(dm)]
            row[k] = round(float(np.median(vals)), 4)
        result.append(row)
    return result


# ---------------------------------------------------------------------------
# Class summary
# ---------------------------------------------------------------------------


def build_class_summary(
    corners: list[dict],
    team_a_metrics: list[dict],
    team_b_metrics: list[dict],
) -> dict[str, dict]:
    """Aggregate corner metrics by class (slow / medium / high).

    Returns:
        Dict keyed by class name; each value has keys:
            corner_count, team_a (CornerMetrics dict), team_b (CornerMetrics dict).
    """
    keys = ["v_min", "exit_speed", "throttle_dist_frac", "brake_point_frac"]
    buckets: dict[str, dict[str, list]] = {
        "slow":   {k: ([], []) for k in keys},  # type: ignore[assignment]
        "medium": {k: ([], []) for k in keys},
        "high":   {k: ([], []) for k in keys},
    }
    # Re-index metrics by corner_number for safe lookup
    a_by_n = {m["corner_number"]: m for m in team_a_metrics}
    b_by_n = {m["corner_number"]: m for m in team_b_metrics}

    for c in corners:
        cls = c["corner_class"]
        cn  = c["corner_number"]
        if cn not in a_by_n or cn not in b_by_n:
            continue
        for k in keys:
            buckets[cls][k][0].append(a_by_n[cn][k])  # type: ignore[index]
            buckets[cls][k][1].append(b_by_n[cn][k])  # type: ignore[index]

    summary: dict[str, dict] = {}
    for cls, data in buckets.items():
        count = len(next(iter(data.values()))[0])
        if count == 0:
            continue
        summary[cls] = {
            "corner_count": count,
            "team_a": {k: round(float(np.median(data[k][0])), 3) for k in keys},
            "team_b": {k: round(float(np.median(data[k][1])), 3) for k in keys},
        }
    return summary


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _rotate_circuit_to_fit(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    target_aspect: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a circuit so its principal axis fills a landscape container.

    Tries all four 90-degree orientations of the PCA principal axis and picks
    the one whose bounding-box aspect ratio is closest to target_aspect
    (width / height).  Identical algorithm to telemetry_compute.
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


# ---------------------------------------------------------------------------
# Plotly figures
# ---------------------------------------------------------------------------

_CLASS_ORDER = ["slow", "medium", "high"]
_CLASS_LABEL = {"slow": "Slow", "medium": "Medium", "high": "High"}


def _dark_layout(**kwargs) -> dict:
    """Base Plotly layout for the app's dark theme."""
    base = {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Space Grotesk, sans-serif", "color": "#fff"},
        "margin": {"l": 60, "r": 30, "t": 50, "b": 50},
    }
    base.update(kwargs)
    return base


def build_v_min_figure(
    corners: list[dict],
    team_a_metrics: list[dict],
    team_a_name: str,
    team_a_color: str,
    team_b_metrics: list[dict],
    team_b_name: str,
    team_b_color: str,
) -> go.Figure:
    """Dumbbell chart: min apex speed per corner, two teams connected by a line."""
    a_by_n = {m["corner_number"]: m for m in team_a_metrics}
    b_by_n = {m["corner_number"]: m for m in team_b_metrics}

    class_rank = {"slow": 0, "medium": 1, "high": 2}
    sorted_corners = sorted(
        corners, key=lambda c: (class_rank[c["corner_class"]], c["corner_number"])
    )

    labels = [
        f"C{c['corner_number']} · {_CLASS_LABEL[c['corner_class']]}"
        for c in sorted_corners
    ]

    # Connecting lines — one segment per corner, separated by None
    line_x: list = []
    line_y: list = []
    for i, c in enumerate(sorted_corners):
        cn = c["corner_number"]
        if cn not in a_by_n or cn not in b_by_n:
            continue
        line_x.extend([a_by_n[cn]["v_min"], b_by_n[cn]["v_min"], None])
        line_y.extend([labels[i], labels[i], None])

    dot_labels_a = [labels[i] for i, c in enumerate(sorted_corners) if c["corner_number"] in a_by_n]
    dot_labels_b = [labels[i] for i, c in enumerate(sorted_corners) if c["corner_number"] in b_by_n]
    dot_x_a = [a_by_n[c["corner_number"]]["v_min"] for c in sorted_corners if c["corner_number"] in a_by_n]
    dot_x_b = [b_by_n[c["corner_number"]]["v_min"] for c in sorted_corners if c["corner_number"] in b_by_n]

    fig = go.Figure()

    # Connecting lines
    fig.add_trace(go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        line={"color": "rgba(255,255,255,0.18)", "width": 2},
        hoverinfo="skip",
        showlegend=False,
    ))

    # Team A dots
    fig.add_trace(go.Scatter(
        x=dot_x_a,
        y=dot_labels_a,
        mode="markers",
        name=team_a_name,
        marker={
            "color": team_a_color,
            "size": 12,
            "line": {"color": "rgba(255,255,255,0.5)", "width": 1},
        },
        hovertemplate="%{x:.1f} km/h<extra>" + team_a_name + "</extra>",
    ))

    # Team B dots
    fig.add_trace(go.Scatter(
        x=dot_x_b,
        y=dot_labels_b,
        mode="markers",
        name=team_b_name,
        marker={
            "color": team_b_color,
            "size": 12,
            "line": {"color": "rgba(255,255,255,0.5)", "width": 1},
        },
        hovertemplate="%{x:.1f} km/h<extra>" + team_b_name + "</extra>",
    ))

    fig.update_layout(
        **_dark_layout(
            title="Min Apex Speed by Corner",
            xaxis={"title": "Min Speed (km/h)"},
            yaxis={"autorange": "reversed", "tickfont": {"size": 10}},
            legend={"orientation": "h", "y": 1.08},
            margin={"l": 120, "r": 30, "t": 60, "b": 50},
        )
    )
    return fig


def build_class_summary_figure(
    summary: dict[str, dict],
    team_a_name: str,
    team_a_color: str,
    team_b_name: str,
    team_b_color: str,
) -> go.Figure:
    """Grouped vertical bar chart: median min speed per corner class."""
    classes = [c for c in _CLASS_ORDER if c in summary]
    x_labels = [_CLASS_LABEL[c] for c in classes]
    v_a = [summary[c]["team_a"]["v_min"] for c in classes]
    v_b = [summary[c]["team_b"]["v_min"] for c in classes]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=team_a_name,
        x=x_labels,
        y=v_a,
        marker_color=team_a_color,
        hovertemplate="%{y:.1f} km/h<extra>" + team_a_name + "</extra>",
    ))
    fig.add_trace(go.Bar(
        name=team_b_name,
        x=x_labels,
        y=v_b,
        marker_color=team_b_color,
        hovertemplate="%{y:.1f} km/h<extra>" + team_b_name + "</extra>",
    ))
    fig.update_layout(
        **_dark_layout(
            title="Median Min Speed by Corner Class",
            barmode="group",
            xaxis={"title": "Corner Class"},
            yaxis={"title": "Median Min Speed (km/h)"},
            legend={"orientation": "h", "y": 1.08},
        )
    )
    return fig


def build_corner_track_map_figure(
    circuit_x: list[float],
    circuit_y: list[float],
    corners: list[dict],
    team_a_metrics: list[dict],
    team_a_name: str,
    team_a_color: str,
    team_b_metrics: list[dict],
    team_b_name: str,
    team_b_color: str,
) -> go.Figure:
    """Circuit outline with apex markers colored by which team has higher v_min."""
    cx = np.array(circuit_x, dtype=float)
    cy = np.array(circuit_y, dtype=float)

    # Rotate so the circuit's principal axis fills a landscape canvas.
    cx, cy = _rotate_circuit_to_fit(cx, cy)

    # Tight axis bounds — 5% padding, matching telemetry track map.
    pad_x = (cx.max() - cx.min()) * 0.05
    pad_y = (cy.max() - cy.min()) * 0.05
    x_range = [float(cx.min() - pad_x), float(cx.max() + pad_x)]
    y_range = [float(cy.min() - pad_y), float(cy.max() + pad_y)]

    n_pts = len(cx)

    a_by_n = {m["corner_number"]: m for m in team_a_metrics}
    b_by_n = {m["corner_number"]: m for m in team_b_metrics}

    fig = go.Figure()

    # Circuit outline
    fig.add_trace(go.Scatter(
        x=cx.tolist(),
        y=cy.tolist(),
        mode="lines",
        line={"color": "rgba(255,255,255,0.25)", "width": 5},
        name="Circuit",
        hoverinfo="skip",
        showlegend=False,
    ))

    # Corner apex markers
    marker_x, marker_y, marker_colors, marker_sizes, hover_texts = [], [], [], [], []
    for c in corners:
        cn = c["corner_number"]
        apex_idx = int(np.clip(round(c["apex_frac"] * (n_pts - 1)), 0, n_pts - 1))
        mx = float(cx[apex_idx])
        my = float(cy[apex_idx])

        va = a_by_n.get(cn, {}).get("v_min")
        vb = b_by_n.get(cn, {}).get("v_min")

        if va is not None and vb is not None:
            delta = round(va - vb, 1)
            if delta > 0.5:
                color = team_a_color
                faster = team_a_name
            elif delta < -0.5:
                color = team_b_color
                faster = team_b_name
            else:
                color = "#ffffff"
                faster = "Equal"
            size = int(np.clip(13 + abs(delta) * 0.4, 13, 26))
            hover = (
                f"C{cn} ({c['corner_class']})<br>"
                f"{team_a_name}: {va:.1f} km/h<br>"
                f"{team_b_name}: {vb:.1f} km/h<br>"
                f"Δ {delta:+.1f} km/h → {faster}"
            )
        else:
            color = "#888888"
            size = 13
            hover = f"C{cn} — no data"

        marker_x.append(mx)
        marker_y.append(my)
        marker_colors.append(color)
        marker_sizes.append(size)
        hover_texts.append(hover)

    fig.add_trace(go.Scatter(
        x=marker_x,
        y=marker_y,
        mode="markers+text",
        text=[str(c["corner_number"]) for c in corners],
        textposition="top center",
        textfont={"size": 9, "color": "rgba(255,255,255,0.7)"},
        marker={
            "color": marker_colors,
            "size": marker_sizes,
            "line": {"color": "rgba(255,255,255,0.4)", "width": 1},
        },
        hovertext=hover_texts,
        hoverinfo="text",
        name="Corners",
        showlegend=False,
    ))

    # Legend proxies for team colors
    for name, color in [(team_a_name, team_a_color), (team_b_name, team_b_color)]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker={"color": color, "size": 10},
            name=f"{name} faster",
        ))

    fig.update_layout(
        **_dark_layout(
            title="Corner Map — Faster Team per Apex",
            xaxis={"visible": False, "scaleanchor": "y", "scaleratio": 1, "range": x_range},
            yaxis={"visible": False, "range": y_range},
            margin={"l": 0, "r": 0, "t": 40, "b": 0},
            legend={"orientation": "h", "y": 1.05, "x": 0.5, "xanchor": "center"},
        )
    )
    return fig
