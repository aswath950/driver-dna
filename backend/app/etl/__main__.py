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

from app.etl import (
    fetch_telemetry,
    hydrate_session,
    refresh_driver_stats,
    refresh_session_key,
    seed_circuit_corners,
    seed_circuits,
)


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
        default=None,
        help=(
            "Preferred season year for FastF1 lookups. Defaults to each "
            "circuit's most recent completed event."
        ),
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

    rsk = sub.add_parser(
        "refresh-session-key",
        help="Re-resolve a session's current OpenF1 session_key and update it in place "
             "(fixes stale keys after OpenF1 renumbering).",
    )
    rsk_grp = rsk.add_mutually_exclusive_group(required=True)
    rsk_grp.add_argument(
        "--session-id", type=int, help="Database session ID to refresh."
    )
    rsk_grp.add_argument(
        "--all", action="store_true",
        help="Scan every session and refresh any whose stored key no longer resolves.",
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
        # A hydrate that wrote nothing is a failure, not a no-op: the weekend
        # was found but every row count came back zero. Exit non-zero so batch
        # callers notice instead of banking a silently-empty seed. A dry run
        # legitimately writes nothing, so it is exempt.
        if not args.dry_run and result.counts and not any(result.counts.values()):
            print(
                f"error: hydrate wrote 0 rows for {args.gp} {args.year} "
                f"(session={args.session or 'all'}) — treating as a failure",
                file=sys.stderr,
            )
            return 1
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
        # Non-zero exit when anything went wrong, so cron/CI and shell callers
        # actually notice (a silent exit 0 is what masked the stale-key failure).
        return 1 if result.errors else 0

    if args.command == "refresh-session-key":
        results = refresh_session_key.run(
            session_id=args.session_id, all_sessions=args.all
        )
        print(json.dumps([r.as_dict() for r in results], indent=2))
        return 1 if any(
            r.status in refresh_session_key._FAILURE_STATUSES for r in results
        ) else 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
