"""Unit tests for ``app.services.corner_compute`` corner-windowing logic.

Focus: the per-corner entry/exit windows must be **disjoint** so two corners
can never share a speed minimum. This is the regression guard for the bug where
close corners (chicanes / esses) reported an identical ``v_min`` in both the
Per-Corner cards and the Track Map, because the old boundary walk was unbounded
and ran straight through the neighbouring corner.

Synthetic-array style mirrors ``test_telemetry_compute.py`` — no DB, no HTTP.
"""

from __future__ import annotations

import json

import numpy as np
import plotly.graph_objects as go
import pytest
from scipy.ndimage import gaussian_filter1d

from app.services import corner_compute as cc

N = cc.N_POINTS  # 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _speed_profile(corner_defs: list[tuple[float, float, float]]) -> np.ndarray:
    """Build a synthetic lap speed array (km/h) from corner definitions.

    Each corner is ``(center_frac, apex_speed_kmh, half_width_frac)`` carved as
    a parabolic dip out of a 300 km/h straight, then lightly smoothed so the
    trace looks like real telemetry.
    """
    x = np.linspace(0.0, 1.0, N)
    speed = np.full(N, 300.0)
    for cf, apex, hw in corner_defs:
        mask = np.abs(x - cf) < hw
        d = (x[mask] - cf) / hw
        speed[mask] = np.minimum(speed[mask], apex + (300.0 - apex) * d**2)
    return gaussian_filter1d(speed, sigma=1.5)


def _preloaded(corner_defs: list[tuple[float, float, float]], length_m: float):
    """FastF1-style preloaded corner records at the given centre fractions."""
    return [
        {"number": i + 1, "letter": "", "distance_m": cf * length_m}
        for i, (cf, _apex, _hw) in enumerate(corner_defs)
    ]


def _windows(corners: list[dict]) -> list[tuple[int, int]]:
    """Convert corner entry/exit fracs back to integer grid indices."""
    return [
        (int(round(c["entry_frac"] * (N - 1))), int(round(c["exit_frac"] * (N - 1))))
        for c in corners
    ]


def _legacy_boundaries(sp_smooth: np.ndarray, apex_idx: int) -> tuple[int, int]:
    """Reproduce the original *unbounded* walk (pre-fix) for a single apex.

    Used to prove the refactor leaves an isolated corner byte-identical when no
    neighbour constrains it (i.e. bounds == full array).
    """
    n = len(sp_smooth)
    v_apex = sp_smooth[apex_idx]
    approach_max = float(sp_smooth[max(0, apex_idx - 50) : apex_idx + 1].max())
    entry_threshold = v_apex + 0.3 * (approach_max - v_apex)
    entry_idx = int(apex_idx)
    while entry_idx > 0 and sp_smooth[entry_idx] < entry_threshold:
        entry_idx -= 1
    exit_max = float(sp_smooth[apex_idx : min(n, apex_idx + 50)].max())
    exit_threshold = v_apex + 0.3 * (exit_max - v_apex)
    exit_idx = int(apex_idx)
    while exit_idx < n - 1 and sp_smooth[exit_idx] < exit_threshold:
        exit_idx += 1
    return entry_idx, exit_idx


def _vmins(speed: np.ndarray, corners: list[dict]) -> list[float]:
    throttle = np.where(speed > 250, 100.0, 0.0)
    brake = np.where(speed < 150, 80.0, 0.0)
    metrics = cc.compute_corner_metrics(
        list(speed), list(throttle), list(brake), corners
    )
    return [m["v_min"] for m in metrics]


# A chicane pair (corners 4 & 5, ~0.06 apart) plus well-separated corners.
CHICANE_DEFS = [
    (0.06, 90, 0.03),
    (0.18, 250, 0.02),
    (0.30, 130, 0.03),
    (0.40, 110, 0.025),  # close pair, shallower
    (0.46, 95, 0.025),   # close pair, deeper
    (0.62, 200, 0.03),
    (0.74, 80, 0.035),
    (0.88, 150, 0.03),
]


# ---------------------------------------------------------------------------
# _territory_bounds — the core invariant
# ---------------------------------------------------------------------------


def test_territory_bounds_are_disjoint_and_contain_apex():
    apexes = [12, 36, 60, 80, 92, 123, 147, 175]
    bounds = cc._territory_bounds(apexes, N)

    assert len(bounds) == len(apexes)
    for apex, (lo, hi) in zip(apexes, bounds):
        assert lo <= apex <= hi
    # Contiguous + non-overlapping: each territory starts after the previous ends.
    for (_, hi_prev), (lo_next, _) in zip(bounds, bounds[1:]):
        assert lo_next > hi_prev


def test_territory_bounds_handle_coincident_apexes():
    # Two corners rounding to the same / adjacent grid index (long circuit).
    apexes = [10, 100, 100, 101, 190]
    bounds = cc._territory_bounds(apexes, N)
    for apex, (lo, hi) in zip(apexes, bounds):
        assert lo <= apex <= hi  # window never collapses past the apex
        assert hi >= lo


# ---------------------------------------------------------------------------
# Regression: close corners must not share v_min
# ---------------------------------------------------------------------------


def test_preloaded_close_corners_have_distinct_vmin():
    speed = _speed_profile(CHICANE_DEFS)
    raw = cc.corners_from_preloaded(_preloaded(CHICANE_DEFS, 5000.0), 5000.0, list(speed))
    corners = cc.classify_corners(raw, list(speed))
    vmins = _vmins(speed, corners)

    # The two close corners (index 3 & 4) used to both report ~115.3.
    assert vmins[3] != vmins[4]
    assert len(set(vmins)) == len(vmins)  # all corners distinct


def test_speed_detection_close_corners_have_distinct_vmin():
    speed = _speed_profile(CHICANE_DEFS)
    raw = cc.detect_corners_from_speed(list(speed))
    corners = cc.classify_corners(raw, list(speed))
    vmins = _vmins(speed, corners)

    assert len(set(vmins)) == len(vmins)


@pytest.mark.parametrize(
    "build",
    [
        lambda s: cc.corners_from_preloaded(_preloaded(CHICANE_DEFS, 5000.0), 5000.0, s),
        lambda s: cc.detect_corners_from_speed(s),
    ],
)
def test_windows_never_overlap(build):
    speed = _speed_profile(CHICANE_DEFS)
    corners = build(list(speed))
    wins = _windows(corners)
    for (_, exit_k), (entry_next, _) in zip(wins, wins[1:]):
        assert entry_next > exit_k, f"window overlap: {wins}"


# ---------------------------------------------------------------------------
# Isolated corner: behaviour unchanged vs the legacy unbounded walk
# ---------------------------------------------------------------------------


def test_isolated_corner_matches_legacy_walk():
    # One corner, far from both ends — neighbours can't constrain it, so the
    # bounded walk must reproduce the original unbounded result exactly.
    speed = _speed_profile([(0.5, 100, 0.04)])
    sp_smooth = gaussian_filter1d(speed, sigma=3)
    apex_idx = int(np.argmin(sp_smooth))

    bounded = cc._corner_boundaries(sp_smooth, apex_idx, 0, N - 1)
    legacy = _legacy_boundaries(sp_smooth, apex_idx)
    assert bounded == legacy

    # Its window brackets the apex, so v_min is the trace's true minimum.
    raw = cc.detect_corners_from_speed(list(speed))
    corners = cc.classify_corners(raw, list(speed))
    vmins = _vmins(speed, corners)
    assert vmins[0] == pytest.approx(float(np.array(speed).min()), abs=0.5)


def test_corner_boundaries_respect_bounds():
    speed = _speed_profile(CHICANE_DEFS)
    sp_smooth = gaussian_filter1d(speed, sigma=3)
    entry, exit_ = cc._corner_boundaries(sp_smooth, apex_idx=92, lower=86, upper=96)
    assert 86 <= entry <= 92 <= exit_ <= 96


# ---------------------------------------------------------------------------
# Straight detection
# ---------------------------------------------------------------------------


def _stadium_outline(
    n_straight: int = 80,
    n_bend: int = 40,
    half_len: float = 500.0,
    radius: float = 120.0,
) -> tuple[list[float], list[float]]:
    """A 'stadium' circuit: two long straights joined by two 180° bends.

    Returns (x, y) outline lists in travel order. The two bends are the only
    high-curvature regions, so they are the only corners; everything else is
    straight.
    """
    L, R = half_len, radius
    top_x = np.linspace(-L, L, n_straight, endpoint=False)
    top_y = np.full(n_straight, R)

    th_r = np.linspace(np.pi / 2, -np.pi / 2, n_bend, endpoint=False)
    rb_x, rb_y = L + R * np.cos(th_r), R * np.sin(th_r)

    bot_x = np.linspace(L, -L, n_straight, endpoint=False)
    bot_y = np.full(n_straight, -R)

    th_l = np.linspace(-np.pi / 2, -3 * np.pi / 2, n_bend, endpoint=False)
    lb_x, lb_y = -L + R * np.cos(th_l), R * np.sin(th_l)

    x = np.concatenate([top_x, rb_x, bot_x, lb_x])
    y = np.concatenate([top_y, rb_y, bot_y, lb_y])
    return x.tolist(), y.tolist()


def test_detect_straights_finds_straights_between_bends():
    x, y = _stadium_outline()
    corners = cc.detect_corners(tuple(x), tuple(y))
    assert len(corners) >= 2  # the two bends

    straights = cc.detect_straights(corners, x, y)
    assert len(straights) == 2  # top + bottom straight

    # Curvature is the discriminator: every straight point sits well below the
    # peak curvature of the bends (a flat-out high-curvature corner stays out).
    _, kappa = cc._curvature(x, y)
    peak = float(kappa.max())
    for st in straights:
        assert st["indices"]  # non-empty
        assert max(kappa[i] for i in st["indices"]) < 0.2 * peak
        # Monotonic, in-range fractions paired 1-to-1 with indices.
        assert len(st["fracs"]) == len(st["indices"])
        assert 0.0 <= st["start_frac"] <= st["end_frac"] <= 1.0


def test_detect_straights_excludes_corner_territory():
    x, y = _stadium_outline()
    corners = cc.detect_corners(tuple(x), tuple(y))
    n = len(x)

    corner_idxs: set[int] = set()
    for c in corners:
        lo = int(round(min(c["entry_frac"], c["exit_frac"]) * (n - 1)))
        hi = int(round(max(c["entry_frac"], c["exit_frac"]) * (n - 1)))
        corner_idxs.update(range(lo, hi + 1))

    straights = cc.detect_straights(corners, x, y)
    for st in straights:
        assert not (set(st["indices"]) & corner_idxs)


def test_detect_straights_empty_inputs():
    x, y = _stadium_outline()
    assert cc.detect_straights([], x, y) == []
    assert cc.detect_straights([{"entry_frac": 0.1, "exit_frac": 0.2}], [], []) == []


# ---------------------------------------------------------------------------
# Straight metrics + speed aggregation
# ---------------------------------------------------------------------------


def test_aggregate_team_speed_single_and_pair():
    a = [100.0] * N
    assert cc.aggregate_team_speed([a]) == a

    b = [200.0] * N
    agg = cc.aggregate_team_speed([a, b])
    assert agg == [150.0] * N  # element-wise median of two


def test_compute_straight_metrics_top_speed_and_location():
    # Increasing speed → Vmax is at the straight's last point.
    team_speed = list(np.linspace(100.0, 300.0, N))
    straight = {
        "straight_number": 1,
        "indices": [10, 11, 12, 13, 14],
        "fracs": [0.0, 0.1, 0.2, 0.3, 0.4],
    }
    [m] = cc.compute_straight_metrics([straight], team_speed)

    assert m["straight_number"] == 1
    assert m["top_speed_index"] == 14
    assert m["top_speed_frac"] == pytest.approx(0.4)
    assert m["top_speed"] == pytest.approx(180.0, abs=1.0)  # 100 + 0.4 * 200


# ---------------------------------------------------------------------------
# Figure builders — smoke tests
# ---------------------------------------------------------------------------


def _stadium_corners_and_metrics():
    x, y = _stadium_outline()
    corners = cc.classify_corners(cc.detect_corners(tuple(x), tuple(y)), _ref_speed())
    speed = _ref_speed()
    throttle = np.where(np.array(speed) > 250, 100.0, 0.0).tolist()
    brake = np.where(np.array(speed) < 150, 80.0, 0.0).tolist()
    metrics = cc.compute_corner_metrics(speed, throttle, brake, corners)
    return x, y, corners, metrics


def _ref_speed() -> list[float]:
    return list(_speed_profile([(0.3, 120, 0.05), (0.8, 90, 0.05)]))


def test_build_straight_performance_figure_smoke():
    x, y = _stadium_outline()
    corners = cc.detect_corners(tuple(x), tuple(y))
    straights = cc.detect_straights(corners, x, y)
    sa = list(np.linspace(120.0, 320.0, N))
    sb = list(np.linspace(110.0, 300.0, N))

    fig = cc.build_straight_performance_figure(
        x, y, straights, sa, "Alpha", "#ff0000", sb, "Bravo", "#0000ff"
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2  # outline + at least one microsector trace
    # Triangle markers for the top-speed points are present.
    symbols = [
        tr.marker.symbol for tr in fig.data if getattr(tr, "marker", None) is not None
    ]
    assert "triangle-up" in symbols
    json.loads(fig.to_json())  # serializable


def test_build_hybrid_map_figure_smoke():
    x, y, corners, metrics = _stadium_corners_and_metrics()
    straights = cc.detect_straights(corners, x, y)
    sa = list(np.linspace(120.0, 320.0, N))
    sb = list(np.linspace(110.0, 300.0, N))

    fig = cc.build_hybrid_map_figure(
        x, y, corners,
        metrics, "Alpha", "#ff0000",
        metrics, "Bravo", "#0000ff",
        straights, sa, sb,
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 3  # outline + corners + straight content
    json.loads(fig.to_json())
