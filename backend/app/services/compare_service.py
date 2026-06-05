"""Compare two drivers' fastest-lap telemetry traces.

Mirrors the chart calculation logic from ``src/viz.py`` (the legacy
Streamlit implementation, canonical reference) so the industrialized
backend produces numerically identical traces and visually-equivalent
Plotly figures.

Data sources differ from Streamlit:
- High-frequency telemetry (speed/throttle/brake samples) still comes from
  OpenF1's ``/car_data`` endpoint — the DB only stores lap aggregates.
- Circuit ``sector_fractions`` and outline ``x``/``y`` come from the
  ``circuits`` table (seeded via ``python -m app.etl seed-circuits``),
  not a JSON file on disk.
- Driver team colours come from ``teams.color_hex`` joined via the driver,
  falling back to a deterministic palette when unset.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.openf1 import OpenF1Client

from app.core.errors import BadRequestError, NotFoundError, UpstreamError
from app.db.models import Circuit, Driver, LapTime, SessionDriver, Team
from app.db.models import Session as SessionRow
from app.db.repositories import circuits as circuits_repo
from app.db.repositories import telemetry as telemetry_repo
from app.services import telemetry_compute as tc

Channel = Literal[
    "Speed", "Throttle", "Brake", "RPM", "nGear", "DRS",
    "TimeDelta", "TrackMap", "SectorTimes",
]

# Fallback palette when a driver/team has no colour. Visually-distinct
# hues; cycles by driver_id parity so two drivers in the same session
# never collide.
_FALLBACK_PALETTE = ["#E10600", "#1E40AF", "#06B6D4", "#FBBF24", "#10B981", "#F472B6"]


def _complementary_color(hex_color: str) -> str:
    """Return the complementary color (hue + 180°) with enforced visibility.

    Clamps lightness to ≥ 0.60 and saturation to ≥ 0.70 so the result is
    always vivid and readable against the app's dark background.
    """
    import colorsys

    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    hue, lgt, sat = colorsys.rgb_to_hls(r, g, b)
    hue_comp = (hue + 0.5) % 1.0
    lgt_comp = max(lgt, 0.60)
    sat_comp = max(sat, 0.70)
    r2, g2, b2 = colorsys.hls_to_rgb(hue_comp, lgt_comp, sat_comp)
    return "#{:02x}{:02x}{:02x}".format(int(r2 * 255), int(g2 * 255), int(b2 * 255))


def _differentiate_colors(color_a: str, color_b: str) -> tuple[str, str]:
    """When both drivers share a team color, assign the complementary hue to driver B."""
    if color_a.lower() == color_b.lower():
        return color_a, _complementary_color(color_b)
    return color_a, color_b


# ---------------------------------------------------------------------------
# DB / session helpers
# ---------------------------------------------------------------------------


async def _resolve_session_key(db: AsyncSession, session_id: int) -> int:
    s = await db.get(SessionRow, session_id)
    if s is None:
        raise NotFoundError("session", session_id)
    if s.openf1_session_key is None:
        raise UpstreamError(
            f"session {session_id} has no openf1_session_key — was it loaded by ETL?"
        )
    return int(s.openf1_session_key)


async def _resolve_driver(
    db: AsyncSession, session_id: int, driver_id: int
) -> tuple[int, str, str]:
    """Return ``(car_number, code, color_hex)`` for a driver in a given session.

    Colour resolution order: ``session_drivers.team → teams.color_hex``,
    then a fallback palette indexed by ``driver_id``.
    """
    row = (
        await db.execute(
            select(SessionDriver.car_number, Driver.code, Team.color_hex)
            .join(Driver, Driver.id == SessionDriver.driver_id)
            .join(Team, Team.id == SessionDriver.team_id)
            .where(
                SessionDriver.session_id == session_id,
                SessionDriver.driver_id == driver_id,
            )
        )
    ).first()
    if row is None:
        raise NotFoundError(
            "session_driver", f"session={session_id} driver={driver_id}"
        )
    color = row.color_hex or _FALLBACK_PALETTE[driver_id % len(_FALLBACK_PALETTE)]
    return int(row.car_number), str(row.code), str(color)


async def _load_circuit(db: AsyncSession, session_id: int) -> Circuit:
    """Load the Circuit row for a session, raising if it's missing geometry."""
    circuit = await circuits_repo.get_for_session(db, session_id)
    if circuit is None:
        raise UpstreamError(
            f"no circuit linked to session {session_id} — was the event hydrated?"
        )
    if not circuit.sector_fractions:
        raise UpstreamError(
            f"circuit {circuit.name!r} has no sector_fractions — "
            "run `python -m app.etl seed-circuits` to seed circuit geometry"
        )
    return circuit


# ---------------------------------------------------------------------------
# OpenF1 fetch — one trip per driver, all channels at once
# ---------------------------------------------------------------------------


def _fastest_lap_row(laps_df: pd.DataFrame, driver_number: int) -> pd.Series | None:
    """Return the fastest non-pit-out lap row for a driver, or None."""
    drv = laps_df[laps_df["driver_number"] == driver_number].copy()
    if "is_pit_out_lap" in drv.columns:
        drv = drv[drv["is_pit_out_lap"] != True]  # noqa: E712
    drv = drv.dropna(subset=["lap_duration"])
    if drv.empty:
        return None
    return drv.sort_values("lap_duration").iloc[0]


def _fetch_fastest_lap_from_openf1(
    *,
    client: OpenF1Client,
    session_key: int,
    driver_number: int,
    laps_df: pd.DataFrame,
) -> tuple[dict | None, pd.DataFrame | None, pd.Series | None]:
    """Fetch one driver's fastest-lap car_data from OpenF1 and run process_car_data.

    Returns ``(processed_dict, raw_car_df, fastest_lap_row)`` so the caller
    can write-through cache the raw samples without a second API call.
    """
    fastest = _fastest_lap_row(laps_df, driver_number)
    if fastest is None:
        return None, None, None

    lap_time = float(fastest["lap_duration"])
    lap_n = int(fastest["lap_number"]) if pd.notna(fastest.get("lap_number")) else None

    date_start = fastest.get("date_start")
    if date_start is None or pd.isna(date_start):
        return None, None, fastest
    ts_start = pd.Timestamp(date_start)
    ts_end = ts_start + pd.Timedelta(seconds=lap_time)

    car = client.get_car_data(
        session_key=session_key,
        driver_number=driver_number,
        date_gte=ts_start.isoformat(),
        date_lte=(ts_end + pd.Timedelta(seconds=0.5)).isoformat(),
    )
    result = tc.process_car_data(car, lap_duration=lap_time, lap_end=ts_end, lap_number=lap_n)
    return result, car, fastest


def _samples_dict(car_df: pd.DataFrame) -> dict:
    """Convert a raw car_data DataFrame to the columnar JSONB format for caching."""
    return {
        "dates": [d.isoformat() for d in car_df["date"]],
        "speed": car_df["speed"].tolist(),
        "throttle": car_df["throttle"].tolist(),
        "brake": car_df["brake"].tolist(),
        "rpm": car_df["rpm"].tolist(),
        "n_gear": car_df["n_gear"].tolist(),
        "drs": car_df["drs"].tolist(),
    }


async def _fastest_lap_from_db(
    db: AsyncSession, session_id: int, driver_id: int
) -> tuple[int, float] | None:
    """Return (lap_number, lap_duration_sec) of the fastest non-pit-out lap
    stored in the ``lap_times`` table, or None if not available."""
    stmt = (
        select(LapTime.lap_number, LapTime.lap_time_ms)
        .where(
            LapTime.session_id == session_id,
            LapTime.driver_id == driver_id,
            LapTime.lap_time_ms.isnot(None),
            LapTime.is_pit_out.is_(False),
        )
        .order_by(LapTime.lap_time_ms.asc())
        .limit(1)
    )
    row = (await db.execute(stmt)).first()
    if row is None:
        return None
    return int(row.lap_number), row.lap_time_ms / 1000.0


async def _fetch_fastest_lap_data(
    db: AsyncSession,
    *,
    session_id: int,
    driver_id: int,
    client: OpenF1Client,
    session_key: int,
    driver_number: int,
    laps_df: pd.DataFrame | None,
) -> dict | None:
    """Cache-first fetch: read from car_telemetry, fall back to OpenF1 with write-through.

    Tries the DB-stored lap_times to find the fastest lap number first so the
    cache lookup can proceed without calling OpenF1's /laps endpoint.  When
    both the lap number and the telemetry samples are cached, zero OpenF1 calls
    are made.  ``laps_df`` may be None on entry; it is populated lazily only
    when an OpenF1 fetch is actually required.
    """
    # 1. Resolve fastest lap number — prefer DB to avoid an OpenF1 call.
    db_lap = await _fastest_lap_from_db(db, session_id, driver_id)
    lap_n: int | None = db_lap[0] if db_lap else None
    cached_lap_duration: float | None = db_lap[1] if db_lap else None

    # 2. Cache hit — reconstruct DataFrame from stored JSONB samples.
    if lap_n is not None:
        cached = await telemetry_repo.get_lap(db, session_id, driver_id, lap_n)
        if cached is not None:
            s = cached.samples
            car_df = pd.DataFrame({
                "date": pd.to_datetime(s["dates"], utc=True),
                "speed": s["speed"],
                "throttle": s["throttle"],
                "brake": s["brake"],
                "rpm": s["rpm"],
                "n_gear": s["n_gear"],
                "drs": s["drs"],
            })
            return tc.process_car_data(
                car_df,
                lap_duration=cached.lap_duration or cached_lap_duration or 0.0,
                lap_number=lap_n,
            )

    # 3. Cache miss — need laps_df from OpenF1 to find the time window.
    if laps_df is None or laps_df.empty:
        return None  # caller must supply a valid laps_df for cache misses

    fastest = _fastest_lap_row(laps_df, driver_number)
    if fastest is None:
        return None
    openf1_lap_n = int(fastest["lap_number"]) if pd.notna(fastest.get("lap_number")) else None
    effective_lap_n = openf1_lap_n

    result, raw_car, _ = _fetch_fastest_lap_from_openf1(
        client=client, session_key=session_key,
        driver_number=driver_number, laps_df=laps_df,
    )

    # 4. Write-through: persist raw samples so the next request is a cache hit.
    if (
        result is not None
        and raw_car is not None
        and effective_lap_n is not None
        and not raw_car.empty
    ):
        try:
            await telemetry_repo.save_lap(
                db,
                session_id=session_id,
                driver_id=driver_id,
                lap_number=effective_lap_n,
                lap_duration=float(fastest["lap_duration"]),
                samples=_samples_dict(raw_car),
            )
            await db.flush()
        except Exception:
            pass  # cache write failure is non-fatal

    return result


# ---------------------------------------------------------------------------
# Compare payload — Speed / Throttle / Brake / RPM / nGear / DRS / TimeDelta
# ---------------------------------------------------------------------------


async def build_compare_payload(
    db: AsyncSession,
    *,
    session_id: int,
    driver_a_id: int,
    driver_b_id: int,
    channel: Channel,
) -> dict:
    """Fetch fastest-lap data for two drivers, build a Plotly figure
    matching the Streamlit implementation, return the payload dict.

    Handles Speed/Throttle/Brake/RPM/nGear/DRS (car-data channels) and
    TimeDelta. For SectorTimes / TrackMap use the dedicated builders below.
    """
    if driver_a_id == driver_b_id:
        raise BadRequestError("driver_a and driver_b must differ")

    session_key = await _resolve_session_key(db, session_id)
    car_a, code_a, color_a = await _resolve_driver(db, session_id, driver_a_id)
    car_b, code_b, color_b = await _resolve_driver(db, session_id, driver_b_id)
    color_a, color_b = _differentiate_colors(color_a, color_b)

    # Only TimeDelta uses sector_fractions overlays; the car-data channels
    # render fine without a seeded circuit, so don't gate them on it.
    circuit = await _load_circuit(db, session_id) if channel == "TimeDelta" else None

    client = OpenF1Client(mode="historical")

    # Phase 1: try the DB cache first — zero OpenF1 calls if fully cached.
    data_a = await _fetch_fastest_lap_data(
        db, session_id=session_id, driver_id=driver_a_id,
        client=client, session_key=session_key,
        driver_number=car_a, laps_df=None,
    )
    data_b = await _fetch_fastest_lap_data(
        db, session_id=session_id, driver_id=driver_b_id,
        client=client, session_key=session_key,
        driver_number=car_b, laps_df=None,
    )

    # Phase 2: any cache miss — fetch laps once from OpenF1 and retry.
    if data_a is None or data_b is None:
        laps_df = client.get_laps(session_key)
        if laps_df.empty:
            raise UpstreamError(f"OpenF1 returned no laps for session_key={session_key}")
        if data_a is None:
            data_a = await _fetch_fastest_lap_data(
                db, session_id=session_id, driver_id=driver_a_id,
                client=client, session_key=session_key,
                driver_number=car_a, laps_df=laps_df,
            )
        if data_b is None:
            data_b = await _fetch_fastest_lap_data(
                db, session_id=session_id, driver_id=driver_b_id,
                client=client, session_key=session_key,
                driver_number=car_b, laps_df=laps_df,
            )

    if data_a is None:
        raise UpstreamError(
            f"OpenF1 returned no usable telemetry for driver_number={car_a} "
            f"session_key={session_key}"
        )
    if data_b is None:
        raise UpstreamError(
            f"OpenF1 returned no usable telemetry for driver_number={car_b} "
            f"session_key={session_key}"
        )

    if channel == "TimeDelta":
        assert circuit is not None  # narrowed above
        fig = tc.build_time_delta_figure(
            data_a, code_a, color_a,
            data_b, code_b, color_b,
            sector_fractions=circuit.sector_fractions,
        )
        # The "trace" field on TimeDelta payloads carries the per-driver
        # cumtime arrays — useful for clients that want raw numbers.
        trace_a = [float(v) for v in data_a["cumtime"]]
        trace_b = [float(v) for v in data_b["cumtime"]]
    else:
        fig = tc.build_channel_figure(
            data_a, code_a, color_a,
            data_b, code_b, color_b,
            channel=channel,
        )
        col = tc.CHANNEL_COLUMN[channel]
        trace_a = [float(v) for v in data_a[col]]
        trace_b = [float(v) for v in data_b[col]]

    return {
        "session_id": session_id,
        "channel": channel,
        "driver_a": {
            "driver_id": driver_a_id,
            "car_number": car_a,
            "code": code_a,
            "fastest_lap_time_sec": data_a["lap_time"],
            "fastest_lap_number": data_a["lap_number"],
            "trace": trace_a,
        },
        "driver_b": {
            "driver_id": driver_b_id,
            "car_number": car_b,
            "code": code_b,
            "fastest_lap_time_sec": data_b["lap_time"],
            "fastest_lap_number": data_b["lap_number"],
            "trace": trace_b,
        },
        "figure_json": fig.to_json(),
    }


# ---------------------------------------------------------------------------
# Sector times payload — cumtime × sector_fractions (port of src/viz)
# ---------------------------------------------------------------------------


def _sec_to_ms(val: float | None) -> int | None:
    if val is None:
        return None
    return int(round(val * 1000))


async def build_sector_times_payload(
    db: AsyncSession,
    *,
    session_id: int,
    driver_a_id: int,
    driver_b_id: int,
) -> dict:
    """Build sector splits and figure from cumtime indexed at sector_fractions.

    Matches ``src/viz._build_sector_times_fig``: splits = cumtime[i1],
    cumtime[i2]-cumtime[i1], cumtime[-1]-cumtime[i2], with i1/i2 derived
    from the circuit's ``sector_fractions``.
    """
    if driver_a_id == driver_b_id:
        raise BadRequestError("driver_a and driver_b must differ")

    session_key = await _resolve_session_key(db, session_id)
    car_a, code_a, color_a = await _resolve_driver(db, session_id, driver_a_id)
    car_b, code_b, color_b = await _resolve_driver(db, session_id, driver_b_id)
    color_a, color_b = _differentiate_colors(color_a, color_b)
    circuit = await _load_circuit(db, session_id)

    client = OpenF1Client(mode="historical")
    data_a = await _fetch_fastest_lap_data(
        db, session_id=session_id, driver_id=driver_a_id,
        client=client, session_key=session_key,
        driver_number=car_a, laps_df=None,
    )
    data_b = await _fetch_fastest_lap_data(
        db, session_id=session_id, driver_id=driver_b_id,
        client=client, session_key=session_key,
        driver_number=car_b, laps_df=None,
    )
    if data_a is None or data_b is None:
        laps_df = client.get_laps(session_key)
        if laps_df.empty:
            raise UpstreamError(f"OpenF1 returned no laps for session_key={session_key}")
        if data_a is None:
            data_a = await _fetch_fastest_lap_data(
                db, session_id=session_id, driver_id=driver_a_id,
                client=client, session_key=session_key,
                driver_number=car_a, laps_df=laps_df,
            )
        if data_b is None:
            data_b = await _fetch_fastest_lap_data(
                db, session_id=session_id, driver_id=driver_b_id,
                client=client, session_key=session_key,
                driver_number=car_b, laps_df=laps_df,
            )
    if data_a is None or data_b is None:
        raise UpstreamError(
            f"OpenF1 returned no usable telemetry for session_key={session_key}"
        )

    fig, splits_a, splits_b = tc.build_sector_times_figure(
        data_a, code_a, color_a,
        data_b, code_b, color_b,
        sector_fractions=circuit.sector_fractions,
    )

    def _splits_dict(
        splits: tuple[float, float, float] | None,
        driver_id: int, code: str, data: dict,
    ) -> dict:
        s1 = s2 = s3 = None
        if splits is not None:
            s1, s2, s3 = (_sec_to_ms(v) for v in splits)
        return {
            "driver_id": driver_id,
            "code": code,
            "lap_number": data["lap_number"],
            "lap_time_ms": _sec_to_ms(data["lap_time"]),
            "sector1_ms": s1,
            "sector2_ms": s2,
            "sector3_ms": s3,
        }

    return {
        "session_id": session_id,
        "driver_a": _splits_dict(splits_a, driver_a_id, code_a, data_a),
        "driver_b": _splits_dict(splits_b, driver_b_id, code_b, data_b),
        "figure_json": fig.to_json(),
    }


# ---------------------------------------------------------------------------
# Track map payload — circuit outline coloured by faster driver per microsector
# ---------------------------------------------------------------------------


async def build_track_map_payload(
    db: AsyncSession,
    *,
    session_id: int,
    driver_a_id: int,
    driver_b_id: int,
) -> dict:
    """Build the track map figure from the circuit's ``x``/``y`` outline
    coloured by which driver's Speed trace is higher in each microsector.

    Replaces the previous per-driver OpenF1 ``/location`` traces.
    """
    if driver_a_id == driver_b_id:
        raise BadRequestError("driver_a and driver_b must differ")

    session_key = await _resolve_session_key(db, session_id)
    car_a, code_a, color_a = await _resolve_driver(db, session_id, driver_a_id)
    car_b, code_b, color_b = await _resolve_driver(db, session_id, driver_b_id)
    color_a, color_b = _differentiate_colors(color_a, color_b)
    circuit = await _load_circuit(db, session_id)

    if not circuit.x or not circuit.y:
        raise UpstreamError(
            f"circuit {circuit.name!r} has no outline geometry — "
            "run `python -m app.etl seed-circuits` to seed circuit x/y"
        )

    client = OpenF1Client(mode="historical")
    data_a = await _fetch_fastest_lap_data(
        db, session_id=session_id, driver_id=driver_a_id,
        client=client, session_key=session_key,
        driver_number=car_a, laps_df=None,
    )
    data_b = await _fetch_fastest_lap_data(
        db, session_id=session_id, driver_id=driver_b_id,
        client=client, session_key=session_key,
        driver_number=car_b, laps_df=None,
    )
    if data_a is None or data_b is None:
        laps_df = client.get_laps(session_key)
        if laps_df.empty:
            raise UpstreamError(f"OpenF1 returned no laps for session_key={session_key}")
        if data_a is None:
            data_a = await _fetch_fastest_lap_data(
                db, session_id=session_id, driver_id=driver_a_id,
                client=client, session_key=session_key,
                driver_number=car_a, laps_df=laps_df,
            )
        if data_b is None:
            data_b = await _fetch_fastest_lap_data(
                db, session_id=session_id, driver_id=driver_b_id,
                client=client, session_key=session_key,
                driver_number=car_b, laps_df=laps_df,
            )
    if data_a is None or data_b is None:
        raise UpstreamError(
            f"OpenF1 returned no usable telemetry for session_key={session_key}"
        )

    fig = tc.build_track_map_figure(
        data_a, code_a, color_a,
        data_b, code_b, color_b,
        circuit_x=circuit.x,
        circuit_y=circuit.y,
        sector_fractions=circuit.sector_fractions,
    )
    if fig is None:  # defensive — we already validated x/y above
        raise UpstreamError(f"could not build track map for circuit {circuit.name!r}")

    return {
        "session_id": session_id,
        "driver_a": {"driver_id": driver_a_id, "code": code_a},
        "driver_b": {"driver_id": driver_b_id, "code": code_b},
        "circuit_x": [float(v) for v in circuit.x],
        "circuit_y": [float(v) for v in circuit.y],
        "figure_json": fig.to_json(),
    }
