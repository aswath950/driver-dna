"""
pipeline.py — Data extraction & feature engineering

Loads F1 session data via FastF1, resamples lap telemetry to a fixed
distance grid, engineers scalar features, and persists to parquet.

Column manifest for dataset.parquet:
  driver, lap_time_seconds, mean_speed, max_speed, min_speed,
  throttle_mean, throttle_std, brake_mean, brake_events,
  gear_changes, steer_std, speed_trace, throttle_trace, brake_trace
"""

import sys
from pathlib import Path

# Guard: must be run inside the project venv to avoid numpy/pandas conflicts
_expected_venv = Path(__file__).parent.parent / ".venv"
if _expected_venv.exists() and not sys.prefix.startswith(str(_expected_venv)):
    print(
        f"\nERROR: Wrong Python interpreter.\n"
        f"  Running: {sys.executable}\n"
        f"  Expected: {_expected_venv}/bin/python\n\n"
        f"  Activate the venv first:\n"
        f"    source .venv/bin/activate\n"
        f"  Then re-run:\n"
        f"    python src/pipeline.py\n",
        file=sys.stderr,
    )
    sys.exit(1)

import fastf1  # noqa: E402
import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402

CACHE_DIR = Path(__file__).parent.parent / "data"
DATASET_PATH = CACHE_DIR / "dataset.parquet"
N_POINTS = 200  # number of evenly spaced distance-based interpolation points

fastf1.Cache.enable_cache(str(CACHE_DIR))


def extract_session_telemetry(
    year: int,
    grand_prix: str,
    session_type: str,
    drivers: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load a FastF1 session and extract engineered lap features for the given drivers.

    Args:
        year:         Season year, e.g. 2024.
        grand_prix:   Event name string, e.g. 'Italian Grand Prix'.
        session_type: 'R', 'Q', 'S', 'SS', 'FP1', 'FP2', 'FP3'.
        drivers:      List of 3-letter driver codes, e.g. ['VER', 'HAM'].
                      Pass None (default) to extract all drivers in the session.

    Returns:
        DataFrame with one row per valid lap. Columns:
            driver, lap_time_seconds, mean_speed, max_speed, min_speed,
            throttle_mean, throttle_std, brake_mean, brake_events,
            gear_changes, steer_std
    """
    session = fastf1.get_session(year, grand_prix, session_type)
    session.load(telemetry=True, laps=True, weather=False)

    if drivers is None:
        drivers = session.laps["Driver"].dropna().unique().tolist()
        print(f"  Found {len(drivers)} drivers: {', '.join(sorted(drivers))}")

    rows = []

    for driver in drivers:
        driver_laps = session.laps.pick_drivers(driver)
        # Exclude in-laps, out-laps, and laps with deleted/missing lap times
        valid_laps = driver_laps.pick_quicklaps()

        for _, lap in valid_laps.iterlaps():
            if pd.isna(lap["LapTime"]):
                continue

            try:
                tel = lap.get_telemetry().add_distance()
            except Exception:
                continue

            if tel.empty or len(tel) < 10:
                continue

            # --- Resample to N_POINTS evenly spaced distance points ---
            dist = tel["Distance"].to_numpy(dtype=float)
            dist_grid = np.linspace(dist[0], dist[-1], N_POINTS)

            channels = ["Speed", "Throttle", "Brake", "nGear", "RPM", "X", "Y"]
            resampled: dict[str, np.ndarray] = {}
            for ch in channels:
                if ch not in tel.columns:
                    resampled[ch] = np.full(N_POINTS, np.nan)
                    continue
                resampled[ch] = np.interp(dist_grid, dist, tel[ch].to_numpy(dtype=float))

            # Steer is optional
            if "Steer" in tel.columns:
                resampled["Steer"] = np.interp(dist_grid, dist, tel["Steer"].to_numpy(dtype=float))
            else:
                resampled["Steer"] = np.full(N_POINTS, np.nan)

            spd = resampled["Speed"]
            thr = resampled["Throttle"]
            brk = resampled["Brake"]
            gear = resampled["nGear"]
            steer = resampled["Steer"]

            # brake_events: count transitions from 0 → >0
            brake_binary = (brk > 0).astype(int)
            brake_events = int(np.diff(brake_binary).clip(min=0).sum())

            # gear_changes: count shifts (any change in gear value)
            gear_changes = int(np.sum(np.diff(np.round(gear)) != 0))

            row = {
                "driver": driver,
                "lap_number": int(lap["LapNumber"]),
                "lap_time_seconds": lap["LapTime"].total_seconds(),
                "mean_speed": float(np.nanmean(spd)),
                "max_speed": float(np.nanmax(spd)),
                "min_speed": float(np.nanmin(spd)),
                "throttle_mean": float(np.nanmean(thr)),
                "throttle_std": float(np.nanstd(thr)),
                "brake_mean": float(np.nanmean(brk)),
                "brake_events": brake_events,
                "gear_changes": gear_changes,
                "steer_std": float(np.nanstd(steer)) if not np.all(np.isnan(steer)) else 0.0,
                # 200-point resampled traces for telemetry overlay plots
                "speed_trace": spd.tolist(),
                "throttle_trace": thr.tolist(),
                "brake_trace": brk.tolist(),
                # Circuit XY coordinates for track map (metres, arbitrary origin)
                "x_trace": resampled["X"].tolist(),
                "y_trace": resampled["Y"].tolist(),
            }
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


META_PATH = CACHE_DIR / "dataset_meta.json"

SESSION_TYPE_LABELS: dict[str, str] = {
    "R":   "Race",
    "Q":   "Qualifying",
    "S":   "Sprint",
    "SS":  "Sprint Shootout",
    "FP1": "Practice 1",
    "FP2": "Practice 2",
    "FP3": "Practice 3",
}


def save_dataset(df: pd.DataFrame, path: Path = DATASET_PATH) -> None:
    """Persist the feature DataFrame to parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} laps → {path}")


def save_meta(year: int, grand_prix: str, session_type: str) -> None:
    """Persist session metadata alongside the dataset for display in the dashboard."""
    import json
    meta = {
        "year": year,
        "grand_prix": grand_prix,
        "session_type": session_type,
        "session_label": SESSION_TYPE_LABELS.get(session_type, session_type),
    }
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"Saved metadata → {META_PATH}")


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the feature DataFrame from parquet."""
    return pd.read_parquet(path)


VALID_SESSION_TYPES = ["Q", "R", "S", "SS", "FP1", "FP2", "FP3"]


def _prompt_inputs() -> tuple[int, str, str, list[str] | None]:
    """Interactively prompt the user for session parameters."""
    print("\n=== Driver DNA — Data Extraction ===\n")

    while True:
        raw = input("Year (e.g. 2024): ").strip()
        if raw.isdigit() and 1950 <= int(raw) <= 2100:
            year = int(raw)
            break
        print("  Enter a valid 4-digit year.")

    grand_prix = input("Grand Prix name (e.g. Italian Grand Prix): ").strip()
    while not grand_prix:
        grand_prix = input("  Cannot be empty. Grand Prix name: ").strip()

    print(f"  Session types: {', '.join(VALID_SESSION_TYPES)}")
    while True:
        session_type = input("Session type: ").strip().upper()
        if session_type in VALID_SESSION_TYPES:
            break
        print(f"  Must be one of: {', '.join(VALID_SESSION_TYPES)}")

    print("  Driver codes are 3-letter abbreviations (e.g. VER, HAM, NOR).")
    print("  Press Enter to extract ALL drivers, or enter specific codes separated by commas:")
    raw_drivers = input("  Drivers [all]: ").strip()
    if raw_drivers:
        drivers: list[str] | None = [d.strip().upper() for d in raw_drivers.split(",") if d.strip()]
    else:
        drivers = None  # signals extract_session_telemetry to use all drivers

    print(f"\nExtracting: {year} {grand_prix} — {session_type}")
    if drivers:
        print(f"Drivers: {', '.join(drivers)}\n")
    else:
        print("Drivers: all\n")
    return year, grand_prix, session_type, drivers


if __name__ == "__main__":
    year, grand_prix, session_type, drivers = _prompt_inputs()

    if DATASET_PATH.exists():
        print(f"  ⚠ {DATASET_PATH} already exists.")
        confirm = input("  Overwrite? [y/N]: ").strip().lower()
        if confirm != "y":
            print("  Aborted — existing dataset kept.")
            raise SystemExit(0)

    df = extract_session_telemetry(
        year=year,
        grand_prix=grand_prix,
        session_type=session_type,
        drivers=drivers,
    )
    print(df)
    save_dataset(df)
    save_meta(year, grand_prix, session_type)
