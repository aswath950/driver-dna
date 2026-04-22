"""
generate_circuits.py — One-off script to build data/circuits.json.

For each F1 Grand Prix in the dashboard dropdown, loads a 2024 race session via
FastF1, extracts the circuit XY outline from the driver with the most completed laps,
resamples to 500 evenly-spaced distance points, and saves to data/circuits.json.

Run once (takes ~10–30 minutes due to FastF1 downloads) then commit circuits.json:

    python src/generate_circuits.py

Re-run whenever a new circuit is added to the GP dropdown in app.py.
"""

import sys
import json
from pathlib import Path

_expected_venv = Path(__file__).parent.parent / ".venv"
if _expected_venv.exists() and not sys.prefix.startswith(str(_expected_venv)):
    print(
        f"\nERROR: Wrong Python interpreter.\n"
        f"  Running: {sys.executable}\n"
        f"  Expected: {_expected_venv}/bin/python\n\n"
        f"  Activate the venv first:\n"
        f"    source .venv/bin/activate\n"
        f"  Then re-run:\n"
        f"    python src/generate_circuits.py\n",
        file=sys.stderr,
    )
    sys.exit(1)

import fastf1  # noqa: E402
import numpy as np  # noqa: E402

CACHE_DIR = Path(__file__).parent.parent / "data"
OUTPUT_PATH = CACHE_DIR / "circuits.json"
N_CIRCUIT_POINTS = 500  # higher than telemetry N_POINTS for a smoother outline

fastf1.Cache.enable_cache(str(CACHE_DIR))

# All GPs in the app.py dropdown, with the reference year to use.
# 2024 is used for all — it was a full 24-race calendar with stable layouts.
GP_LIST = [
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

REFERENCE_YEAR = 2024


def extract_circuit_xy(year: int, gp_name: str) -> dict | None:
    """
    Load a race session and extract the circuit XY outline.

    Returns {"x": [...], "y": [...]} with N_CIRCUIT_POINTS values each,
    or None if the session cannot be loaded or telemetry is unavailable.
    """
    print(f"  Loading {year} {gp_name} ...", end="", flush=True)
    try:
        session = fastf1.get_session(year, gp_name, "R")
        session.load(telemetry=True, laps=True, weather=False)
    except Exception as exc:
        print(f" FAILED ({exc})")
        return None

    laps = session.laps
    if laps is None or laps.empty:
        print(" FAILED (no laps)")
        return None

    # Pick the driver who completed the most laps — most likely to have a clean full circuit
    driver_lap_counts = laps.groupby("Driver")["LapNumber"].count()
    if driver_lap_counts.empty:
        print(" FAILED (no drivers)")
        return None
    best_driver = driver_lap_counts.idxmax()

    # Use that driver's fastest lap for a clean, representative circuit trace
    driver_laps = laps.pick_drivers(best_driver).pick_quicklaps()
    if driver_laps.empty:
        print(" FAILED (no quick laps)")
        return None

    fastest_lap = driver_laps.sort_values("LapTime").iloc[0]
    try:
        tel = fastest_lap.get_telemetry().add_distance()
    except Exception as exc:
        print(f" FAILED (telemetry: {exc})")
        return None

    if tel.empty or "X" not in tel.columns or "Y" not in tel.columns:
        print(" FAILED (no X/Y in telemetry)")
        return None

    dist = tel["Distance"].to_numpy(dtype=float)
    x_raw = tel["X"].to_numpy(dtype=float)
    y_raw = tel["Y"].to_numpy(dtype=float)

    dist_grid = np.linspace(dist[0], dist[-1], N_CIRCUIT_POINTS)
    x_out = np.interp(dist_grid, dist, x_raw)
    y_out = np.interp(dist_grid, dist, y_raw)

    print(f" OK ({N_CIRCUIT_POINTS} pts, driver={best_driver})")
    return {"x": x_out.tolist(), "y": y_out.tolist()}


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing data so a partial run can be resumed without re-downloading
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            circuits: dict = json.load(f)
        print(f"Loaded existing circuits.json ({len(circuits)} entries). Skipping already-extracted circuits.")
    else:
        circuits = {}

    failed = []
    for gp_name in GP_LIST:
        if gp_name in circuits:
            print(f"  Skipping {gp_name} (already in circuits.json)")
            continue
        result = extract_circuit_xy(REFERENCE_YEAR, gp_name)
        if result is not None:
            circuits[gp_name] = result
            # Save after each circuit so progress is preserved on interruption
            with open(OUTPUT_PATH, "w") as f:
                json.dump(circuits, f)
        else:
            failed.append(gp_name)

    print(f"\nDone. {len(circuits)} circuits saved to {OUTPUT_PATH}")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)}")
        print("Re-run this script to retry failed circuits, or check the FastF1 cache.")


if __name__ == "__main__":
    main()
