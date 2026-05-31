"""ETL CLI dispatcher.

    python -m app.etl hydrate --year 2024 --gp "Monaco" --session R
    python -m app.etl hydrate --year 2024 --gp "Monaco"             # all sessions
    python -m app.etl hydrate --year 2024 --gp "Monaco" --dry-run
    python -m app.etl refresh-stats --season 2024
    python -m app.etl refresh-stats --all-seasons
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from app.etl import hydrate_session, refresh_driver_stats


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

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
