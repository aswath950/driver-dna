"""Unit tests for ``app.services.telemetry_compute``.

Two categories:

1. **Algorithm invariants** on synthetic car_data: cumtime anchoring to
   ``lap_duration``, ``np.unique`` dedup at zero-speed sections, sector-time
   accounting, lengths.

2. **Byte-for-byte parity** against the canonical Streamlit implementation
   (``src/viz._fetch_fastest_lap_all_openf1``). We patch its internal
   ``OpenF1Client`` so both code paths run the same post-fetch logic over an
   identical synthetic input, then assert ``np.testing.assert_allclose``.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.services import telemetry_compute as tc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_car_data(
    *,
    n_samples: int = 280,
    dt_sec: float = 0.27,
    base_speed_kmh: float = 200.0,
    zero_speed_window: tuple[int, int] | None = None,
    start: str = "2099-04-01T15:00:00+00:00",
) -> pd.DataFrame:
    """Build a fake OpenF1 car_data DataFrame.

    The speed varies gently so distance is mostly strictly increasing. When
    ``zero_speed_window`` is given, samples in that index range are set to 0
    so the distance proxy has a flat segment — exercising the np.unique dedup.
    """
    times = pd.date_range(start=start, periods=n_samples, freq=f"{int(dt_sec * 1000)}ms")
    speed = base_speed_kmh + 30.0 * np.sin(np.linspace(0, 4 * np.pi, n_samples))
    throttle = 50.0 + 50.0 * np.sin(np.linspace(0, 6 * np.pi, n_samples))
    brake = np.clip(-np.sin(np.linspace(0, 6 * np.pi, n_samples)) * 100, 0, 100)
    rpm = 9000 + 2000 * np.sin(np.linspace(0, 4 * np.pi, n_samples))
    n_gear = np.clip((speed / 50).astype(int), 1, 8)
    drs = np.zeros(n_samples)

    if zero_speed_window is not None:
        lo, hi = zero_speed_window
        speed[lo:hi] = 0.0

    return pd.DataFrame({
        "date": times,
        "speed": speed,
        "throttle": throttle,
        "brake": brake,
        "rpm": rpm,
        "n_gear": n_gear,
        "drs": drs,
    })


# ---------------------------------------------------------------------------
# Algorithm invariants
# ---------------------------------------------------------------------------


def test_n_points_is_400() -> None:
    # The resample grid was raised 200 → 400 for finer corner detail.
    assert tc.N_POINTS == 400


def test_corner_compute_n_points_matches_telemetry() -> None:
    # corner_compute duplicates the constant; apex fractions index the same grid,
    # so the two MUST stay equal. This guards against silent drift.
    from app.services import corner_compute as cc

    assert cc.N_POINTS == tc.N_POINTS


def test_process_car_data_returns_n_points_arrays() -> None:
    car = _synthetic_car_data()
    out = tc.process_car_data(car, lap_duration=75.0)
    assert out is not None
    for key in ("speed", "throttle", "brake", "rpm", "n_gear", "drs", "cumtime"):
        assert out[key].shape == (tc.N_POINTS,), f"{key} wrong shape"
    assert out["lap_time"] == 75.0


def test_process_car_data_cumtime_anchored_to_lap_duration() -> None:
    """cumtime must end exactly at lap_duration after the rescaling step."""
    car = _synthetic_car_data()
    out = tc.process_car_data(car, lap_duration=82.345)
    assert out is not None
    assert out["cumtime"][0] == pytest.approx(0.0, abs=1e-9)
    assert out["cumtime"][-1] == pytest.approx(82.345, rel=1e-6)


def test_process_car_data_dedup_eliminates_zero_speed_spike() -> None:
    """A flat (zero-speed) distance segment must not produce a cumtime spike.

    Without np.unique dedup, np.interp on a non-strictly-increasing x would
    return one of the duplicates' y values arbitrarily, often producing a
    visible discontinuity. With dedup, cumtime stays monotonically
    non-decreasing.
    """
    car = _synthetic_car_data(zero_speed_window=(40, 80))
    out = tc.process_car_data(car, lap_duration=70.0)
    assert out is not None
    diffs = np.diff(out["cumtime"])
    assert (diffs >= -1e-9).all(), "cumtime must be monotonically non-decreasing"


def test_process_car_data_clips_to_lap_end() -> None:
    """Samples past lap_end must be dropped so dist/cumtime stay faithful."""
    car = _synthetic_car_data(n_samples=300, dt_sec=0.27)
    # lap_end at ~the midpoint of the synthetic samples
    lap_end = car["date"].iloc[150]
    out = tc.process_car_data(car, lap_duration=40.0, lap_end=lap_end)
    assert out is not None
    # cumtime anchored to lap_duration regardless of clipping
    assert out["cumtime"][-1] == pytest.approx(40.0, rel=1e-6)


def test_process_car_data_returns_none_on_too_few_samples() -> None:
    car = _synthetic_car_data(n_samples=1)
    assert tc.process_car_data(car, lap_duration=75.0) is None


def test_compute_sector_times_partitions_cumtime() -> None:
    cumtime = np.linspace(0.0, 90.0, tc.N_POINTS)
    splits = tc.compute_sector_times(cumtime, [0.3, 0.7])
    assert splits is not None
    s1, s2, s3 = splits
    # cumtime is linear, so sectors are proportional to fraction widths
    assert s1 + s2 + s3 == pytest.approx(90.0, rel=1e-6)
    assert s1 == pytest.approx(90.0 * round(0.3 * (tc.N_POINTS - 1)) / (tc.N_POINTS - 1), rel=1e-6)


# ---------------------------------------------------------------------------
# Parity with src/viz (canonical Streamlit implementation)
# ---------------------------------------------------------------------------


def _import_src_viz():
    """Import ``src.viz`` for the parity test.

    ``src/features.py`` pulls in Streamlit for its cache decorators, which the
    backend venv doesn't have. We inject a tiny fake ``streamlit`` module that
    no-ops the decorators so the rest of the module loads cleanly. This keeps
    the parity test honest — we run the *real* ``src/viz`` algorithm, not a
    copy that could silently drift.
    """
    import sys
    import types

    if "streamlit" not in sys.modules:
        fake = types.ModuleType("streamlit")

        def _noop_decorator(*dargs, **dkwargs):
            # Support both @st.cache_data and @st.cache_data(...)
            if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
                return dargs[0]
            def _wrap(fn):
                return fn
            return _wrap

        fake.cache_data = _noop_decorator
        fake.cache_resource = _noop_decorator
        sys.modules["streamlit"] = fake

    import app  # ensure src/ is on sys.path  # noqa: F401
    import src.viz as viz  # noqa: WPS433
    return viz


def test_parity_with_src_viz_fetch_fastest_lap_all() -> None:
    """Run both telemetry_compute.process_car_data and
    src/viz._fetch_fastest_lap_all_openf1 over the same mocked car_data
    and assert byte-for-byte equality.

    src/viz constructs its own OpenF1Client and lap window from a laps_df,
    so we patch the client and pass a single-lap synthetic laps_df.
    """
    viz = _import_src_viz()
    _fetch_fastest_lap_all_openf1 = viz._fetch_fastest_lap_all_openf1

    lap_duration = 80.123
    lap_start = pd.Timestamp("2099-04-01T15:00:00+00:00")
    car = _synthetic_car_data(
        n_samples=300, dt_sec=0.27, start=lap_start.isoformat()
    )

    laps_df = pd.DataFrame([{
        "driver_number": 1,
        "lap_number": 5,
        "lap_duration": lap_duration,
        "date_start": lap_start,
        "is_pit_out_lap": False,
    }])

    # Patch the OpenF1Client used inside src/viz so it returns our synthetic
    # car_data verbatim. The post-fetch processing is the part under test.
    with patch.object(viz, "OpenF1Client") as MockClient:
        MockClient.return_value.get_car_data.return_value = car.copy()
        viz_out = _fetch_fastest_lap_all_openf1(
            session_key=999,
            driver_number=1,
            laps_df=laps_df,
        )

    assert viz_out is not None

    # Replicate src/viz's clipping: ts_end = lap_start + lap_duration
    lap_end = lap_start + pd.Timedelta(seconds=lap_duration)
    tc_out = tc.process_car_data(
        car.copy(), lap_duration=lap_duration, lap_end=lap_end, lap_number=5,
    )
    assert tc_out is not None

    for key in ("speed", "throttle", "brake", "cumtime"):
        np.testing.assert_allclose(
            tc_out[key], viz_out[key],
            rtol=1e-12, atol=1e-12,
            err_msg=f"divergence in {key}",
        )
    assert tc_out["lap_time"] == viz_out["lap_time"]
    assert tc_out["lap_number"] == viz_out["lap_number"]


# ---------------------------------------------------------------------------
# Figure builders — smoke tests (just check no exceptions + structural bits)
# ---------------------------------------------------------------------------


def _two_drivers_data() -> tuple[dict, dict]:
    a = tc.process_car_data(
        _synthetic_car_data(start="2099-04-01T15:00:00+00:00"),
        lap_duration=80.0, lap_number=5,
    )
    b = tc.process_car_data(
        _synthetic_car_data(base_speed_kmh=205.0, start="2099-04-01T15:00:00+00:00"),
        lap_duration=80.5, lap_number=6,
    )
    assert a is not None and b is not None
    return a, b


def test_build_time_delta_figure_includes_sector_overlays() -> None:
    a, b = _two_drivers_data()
    fig = tc.build_time_delta_figure(
        a, "VER", "#1E40AF",
        b, "HAM", "#06B6D4",
        sector_fractions=[0.33, 0.66],
    )
    # The three sector vlines + their labels are rendered as layout shapes /
    # annotations. We assert at least 3 vlines (S1 at 0, S2 at 33%, S3 at 66%).
    shapes = fig.layout.shapes or ()
    vlines = [s for s in shapes if getattr(s, "type", None) == "line"]
    assert len(vlines) >= 3, f"expected ≥3 sector vlines, got {len(vlines)}"


def test_build_sector_times_figure_returns_splits() -> None:
    a, b = _two_drivers_data()
    fig, splits_a, splits_b = tc.build_sector_times_figure(
        a, "VER", "#1E40AF",
        b, "HAM", "#06B6D4",
        sector_fractions=[0.33, 0.66],
    )
    assert splits_a is not None and splits_b is not None
    # Sum of sector times must equal final cumtime (== lap_duration after anchor)
    assert sum(splits_a) == pytest.approx(a["cumtime"][-1], rel=1e-6)
    assert sum(splits_b) == pytest.approx(b["cumtime"][-1], rel=1e-6)
    assert len(fig.data) == 2  # one bar trace per driver


def test_build_sector_times_figure_fallback_when_no_fractions() -> None:
    a, b = _two_drivers_data()
    fig, sa, sb = tc.build_sector_times_figure(
        a, "VER", "#1E40AF",
        b, "HAM", "#06B6D4",
        sector_fractions=None,
    )
    assert sa is None and sb is None
    # The fallback figure has one text annotation and no data traces.
    assert len(fig.data) == 0


def test_build_track_map_figure_returns_none_without_geometry() -> None:
    a, b = _two_drivers_data()
    fig = tc.build_track_map_figure(
        a, "VER", "#1E40AF",
        b, "HAM", "#06B6D4",
        circuit_x=None, circuit_y=None,
        sector_fractions=[0.33, 0.66],
    )
    assert fig is None


def test_build_track_map_figure_produces_segments_and_sector_markers() -> None:
    a, b = _two_drivers_data()
    # Synthetic circuit outline — 500 points
    theta = np.linspace(0, 2 * np.pi, 500)
    circuit_x = (1000 * np.cos(theta)).tolist()
    circuit_y = (1000 * np.sin(theta)).tolist()

    fig = tc.build_track_map_figure(
        a, "VER", "#1E40AF",
        b, "HAM", "#06B6D4",
        circuit_x=circuit_x, circuit_y=circuit_y,
        sector_fractions=[0.33, 0.66],
    )
    assert fig is not None
    # Background outline + at least one winner segment + 3 sector markers
    assert len(fig.data) >= 1 + 1 + 3


def test_build_channel_figure_two_traces() -> None:
    a, b = _two_drivers_data()
    fig = tc.build_channel_figure(
        a, "VER", "#1E40AF",
        b, "HAM", "#06B6D4",
        channel="Speed",
    )
    assert len(fig.data) == 2
    assert len(fig.data[0].y) == tc.N_POINTS


# ---------------------------------------------------------------------------
# Speed + Time-delta combined figure
# ---------------------------------------------------------------------------


def test_build_speed_time_delta_figure_stacks_both_panels() -> None:
    a, b = _two_drivers_data()
    fig = tc.build_speed_time_delta_figure(
        a, "VER", "#1E40AF",
        b, "HAM", "#06B6D4",
    )
    # Two stacked panels → a second y-axis exists.
    assert fig.layout.yaxis2 is not None
    y_titles = {fig.layout.yaxis.title.text, fig.layout.yaxis2.title.text}
    assert "Speed (km/h)" in y_titles
    assert any(t and t.startswith("Gap (s)") for t in y_titles)

    # Top panel: one speed line per driver, named with the acronym + lap.
    speed_traces = [t for t in fig.data if t.name and t.name.startswith(("VER (", "HAM ("))]
    assert len(speed_traces) == 2
    # Bottom panel: the two signed delta fills (positive / negative areas).
    fills = [t for t in fig.data if getattr(t, "fill", None) == "tozeroy"]
    assert len(fills) == 2

    # Momentum wash → at least one background rectangle on the speed panel.
    rects = [s for s in (fig.layout.shapes or ()) if s.type == "rect"]
    assert len(rects) >= 1

    # Hovering either panel must surface BOTH panels' data at the same x. That
    # only works when both panels live on one *shared* x-axis (a grid-"coupled"
    # column), not the matched-but-separate axes make_subplots would create —
    # plus unified hover and axis-wide hoversubplots. This is the exact config
    # verified to cross panels; regressing any part silently breaks the hover.
    assert fig.layout.hovermode == "x unified"
    assert fig.layout.hoversubplots == "axis"
    assert (fig.layout.grid.rows, fig.layout.grid.columns) == (2, 1)
    assert fig.layout.grid.pattern == "coupled"
    # Every trace is on the single shared x-axis; speed on "y", gap fills on "y2".
    assert {t.xaxis for t in fig.data} == {"x"}
    assert fig.data[0].yaxis == "y"
    assert {t.yaxis for t in fig.data if getattr(t, "fill", None) == "tozeroy"} == {"y2"}


def test_build_speed_time_delta_figure_overlays_sectors_and_turns() -> None:
    a, b = _two_drivers_data()
    fig = tc.build_speed_time_delta_figure(
        a, "VER", "#1E40AF",
        b, "HAM", "#06B6D4",
        sector_fractions=[0.33, 0.66],
        corners=_sample_corners(),
        circuit_length_m=4000.0,
    )
    # Three sector boundaries drawn through both panels → ≥6 dotted vlines.
    vlines = [s for s in (fig.layout.shapes or ()) if s.type == "line"]
    assert len(vlines) >= 6, f"expected ≥6 sector vlines, got {len(vlines)}"

    # Turn ticks land on the single shared x-axis, relabelled T1/T2A/T3.
    assert fig.layout.xaxis.ticktext == ("T1", "T2A", "T3")
    assert fig.layout.xaxis.title.text == "Track Position (turn)"


def test_momentum_bands_tile_lap_and_merge_noise() -> None:
    x_pct = np.linspace(0.0, 100.0, tc.N_POINTS)
    # Gap rises (A gaining) over the first half then falls (B gaining), with
    # rapid low-amplitude wiggles that must be smoothed and merged away.
    half = tc.N_POINTS // 2
    base = np.concatenate([
        np.linspace(0.0, 0.8, half),
        np.linspace(0.8, -0.8, tc.N_POINTS - half),
    ])
    noisy = base + 0.02 * np.sin(np.linspace(0.0, 80.0, tc.N_POINTS))
    bands = tc._momentum_bands(noisy, x_pct)

    # Bands tile [0, 100] contiguously with no gaps or overlaps.
    assert bands[0][0] == pytest.approx(0.0)
    assert bands[-1][1] == pytest.approx(100.0)
    for (_, x1, _), (x0_next, _, _) in zip(bands, bands[1:]):
        assert x1 == pytest.approx(x0_next)
    # Both drivers' momentum is represented and labels are valid.
    labels = [lab for *_, lab in bands]
    assert set(labels) <= {"a", "b"}
    assert "a" in labels and "b" in labels
    # No barcode: every emitted band is wide enough to read.
    assert all((x1 - x0) >= 4.0 for x0, x1, _ in bands), bands


# ---------------------------------------------------------------------------
# Corner axis ticks + km/h labelling
# ---------------------------------------------------------------------------


def _sample_corners() -> list[dict]:
    # 4000 m lap with turns at 0%, 25%, 50% and (out-of-range) past the line.
    return [
        {"number": 1, "letter": "", "distance_m": 0.0},
        {"number": 2, "letter": "A", "distance_m": 1000.0},
        {"number": 3, "letter": "nan", "distance_m": 2000.0},
    ]


def test_corner_axis_ticks_maps_to_axis_max() -> None:
    ticks = tc.corner_axis_ticks(_sample_corners(), 4000.0, axis_max=tc.N_POINTS - 1)
    assert ticks is not None
    tickvals, ticktext = ticks
    axis_max = tc.N_POINTS - 1
    # corners at 0% / 25% / 50% of the lap map to those fractions of axis_max.
    assert tickvals == [0.0, round(0.25 * axis_max, 3), round(0.5 * axis_max, 3)]
    # Real letters are kept; "nan"/empty letters are dropped.
    assert ticktext == ["T1", "T2A", "T3"]


def test_corner_axis_ticks_returns_none_without_data() -> None:
    assert tc.corner_axis_ticks(None, 4000.0, axis_max=199) is None
    assert tc.corner_axis_ticks(_sample_corners(), None, axis_max=199) is None
    assert tc.corner_axis_ticks([], 4000.0, axis_max=199) is None


def test_corner_axis_ticks_skips_out_of_range_corners() -> None:
    corners = [{"number": 9, "letter": "", "distance_m": 5000.0}]  # past S/F
    assert tc.corner_axis_ticks(corners, 4000.0, axis_max=199) is None


def test_build_channel_figure_speed_labels_kmh() -> None:
    a, b = _two_drivers_data()
    fig = tc.build_channel_figure(
        a, "VER", "#1E40AF",
        b, "HAM", "#06B6D4",
        channel="Speed",
    )
    assert fig.layout.yaxis.title.text == "Speed (km/h)"
    assert all("km/h" in (t.hovertemplate or "") for t in fig.data)


def test_build_channel_figure_non_speed_has_no_kmh() -> None:
    a, b = _two_drivers_data()
    fig = tc.build_channel_figure(
        a, "VER", "#1E40AF",
        b, "HAM", "#06B6D4",
        channel="Throttle",
    )
    assert fig.layout.yaxis.title.text == "Throttle"
    assert all(t.hovertemplate is None for t in fig.data)


def test_build_channel_figure_uses_turn_ticks_when_corners_given() -> None:
    a, b = _two_drivers_data()
    fig = tc.build_channel_figure(
        a, "VER", "#1E40AF",
        b, "HAM", "#06B6D4",
        channel="Speed",
        corners=_sample_corners(),
        circuit_length_m=4000.0,
    )
    assert fig.layout.xaxis.title.text == "Track Position (turn)"
    assert list(fig.layout.xaxis.ticktext) == ["T1", "T2A", "T3"]
    # One guide vline per turn.
    vlines = [s for s in (fig.layout.shapes or ()) if getattr(s, "type", None) == "line"]
    assert len(vlines) == 3


def test_build_channel_figure_falls_back_without_corners() -> None:
    a, b = _two_drivers_data()
    fig = tc.build_channel_figure(
        a, "VER", "#1E40AF",
        b, "HAM", "#06B6D4",
        channel="Speed",
    )
    assert fig.layout.xaxis.title.text == f"Normalised Distance (0–{tc.N_POINTS} points)"


def test_build_time_delta_figure_uses_turn_ticks_when_corners_given() -> None:
    a, b = _two_drivers_data()
    fig = tc.build_time_delta_figure(
        a, "VER", "#1E40AF",
        b, "HAM", "#06B6D4",
        sector_fractions=[0.33, 0.66],
        corners=_sample_corners(),
        circuit_length_m=4000.0,
    )
    assert fig.layout.xaxis.title.text == "Track Position (turn)"
    assert list(fig.layout.xaxis.ticktext) == ["T1", "T2A", "T3"]
