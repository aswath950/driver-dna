"""ETL CLI dispatcher.

    python -m app.etl hydrate --year 2024 --gp "Monaco" --session R
    python -m app.etl hydrate --year 2024 --gp "Monaco"             # all sessions
    python -m app.etl hydrate --year 2024 --gp "Monaco" --dry-run
    python -m app.etl refresh-stats --season 2024
    python -m app.etl refresh-stats --all-seasons
    python -m app.etl seed-circuits
    python -m app.etl seed-circuits --path /custom/path/to/circuits.json
    python -m app.etl seed-circuit-corners
    python -m app.etl seed-circuit-corners --year 2024
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from app.etl import fetch_telemetry, hydrate_session, refresh_driver_stats, seed_circuit_corners, seed_circuits


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m app.etl")
    sub = p.add_subparsers(dest="command", required=True)

    hp = sub.add_parser("hydrate", help="Hydrate one or all sessions for a race weekend.")
    hp.add_argument("--year", type=int, required=True)
    hp.add_argument("--gp", required=True, help='Grand Prix name, e.g. "Monaco"')
    hp.add_argument(
        "--session",
        default=None,
        help="Session type: FP1/FP2/FP3/Q/SQ/S/R. Omit for all sessions in the weekend.",
    )
    hp.add_argument("--dry-run", action="store_true")

    rs = sub.add_parser("refresh-stats", help="Recompute driver_stats from race_results.")
    grp = rs.add_mutually_exclusive_group(required=True)
    grp.add_argument("--season", type=int, help="Refresh stats for one season year.")
    grp.add_argument("--all-seasons", action="store_true", help="Refresh every season.")

    sc = sub.add_parser(
        "seed-circuits", help="Upsert circuit geometry from data/circuits.json."
    )
    sc.add_argument(
        "--path",
        default=None,
        help="Override path to circuits.json (defaults to repo data/circuits.json).",
    )

    scc = sub.add_parser(
        "seed-circuit-corners",
        help="Fetch official corner data from FastF1 and store in circuits.corners.",
    )
    scc.add_argument(
        "--year",
        type=int,
        default=2024,
        help="Season year to look up representative events for each circuit (default: 2024).",
    )

    ft = sub.add_parser(
        "fetch-telemetry",
        help="Download and cache all car telemetry for a session from OpenF1.",
    )
    ft.add_argument(
        "--session-id",
        type=int,
        required=True,
        help="Database session ID (e.g. 73).",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "hydrate":
        result = hydrate_session.run(
            year=args.year,
            grand_prix=args.gp,
            session_type=args.session,
            dry_run=args.dry_run,
        )
        print(json.dumps(result.as_dict(), indent=2, default=str))
        return 0

    if args.command == "refresh-stats":
        year = None if args.all_seasons else args.season
        result = refresh_driver_stats.run(season_year=year)
        print(json.dumps(result.as_dict(), indent=2))
        return 0

    if args.command == "seed-circuits":
        path = Path(args.path) if args.path else None
        result = seed_circuits.run(path=path)
        print(json.dumps(result.as_dict(), indent=2))
        return 0

    if args.command == "seed-circuit-corners":
        result = seed_circuit_corners.run(year=args.year)
        print(json.dumps(result.as_dict(), indent=2))
        return 0

    if args.command == "fetch-telemetry":
        result = fetch_telemetry.run(session_id=args.session_id)
        print(json.dumps(result.as_dict(), indent=2))
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
