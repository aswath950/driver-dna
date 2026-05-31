#!/usr/bin/env bash
# Seed a fresh DB with a handful of representative race weekends so the
# landing page + analytics endpoints have data to display immediately
# after deploy.
#
# Run from the backend container/host:
#
#   bash scripts/hydrate_initial_data.sh             # all races below
#   bash scripts/hydrate_initial_data.sh "Monaco Grand Prix"
#
# Idempotent — safe to re-run.

set -euo pipefail

YEAR="${YEAR:-2024}"

RACES=(
  "Bahrain Grand Prix"
  "Monaco Grand Prix"
  "British Grand Prix"
  "Italian Grand Prix"
  "Las Vegas Grand Prix"
)

if [[ $# -gt 0 ]]; then
  RACES=("$@")
fi

cd "$(dirname "$0")/.."

for gp in "${RACES[@]}"; do
  echo "==> hydrating $YEAR — $gp (R)"
  python -m app.etl hydrate --year "$YEAR" --gp "$gp" --session R || {
    echo "    (skipped: $gp — OpenF1 returned no data)"
    continue
  }
done

echo "==> recomputing driver_stats for $YEAR"
python -m app.etl refresh-stats --season "$YEAR"

echo "done."
