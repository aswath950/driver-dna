#!/usr/bin/env bash
# Seed a fresh DB with the first 11 rounds of the 2026 season so the
# landing page + analytics endpoints have data to display immediately
# after deploy.
#
# Rounds 4-5 (Bahrain, Saudi Arabia) were cancelled by F1 due to the
# Middle East conflict, so this list picks up at Miami (round 4 as
# actually run). Note OpenF1 calls the June Spain race "Barcelona Grand
# Prix" — "Spanish Grand Prix" is reserved for the separate Madrid race
# later in the season.
#
# Run from the backend container/host:
#
#   bash scripts/hydrate_initial_data.sh             # all races below
#   bash scripts/hydrate_initial_data.sh "Monaco Grand Prix"
#   DELAY=15 bash scripts/hydrate_initial_data.sh    # slower, if still rate-limited
#
# Idempotent — safe to re-run.

set -euo pipefail

YEAR="${YEAR:-2026}"

# Seconds to wait between races. OpenF1 rate-limits aggressively and each race
# costs ~6 API calls, so hydrating the full list back-to-back reliably trips
# HTTP 429. That surfaces two ways, neither obvious: a bogus "OpenF1 returned no
# sessions" failure, or — worse — a race that writes 0 rows and still exits 0.
# Set DELAY=0 to disable.
DELAY="${DELAY:-10}"

RACES=(
  "Australian Grand Prix"
  "Chinese Grand Prix"
  "Japanese Grand Prix"
  "Miami Grand Prix"
  "Canadian Grand Prix"
  "Monaco Grand Prix"
  "Barcelona Grand Prix"
  "Austrian Grand Prix"
  "British Grand Prix"
  "Belgian Grand Prix"
  "Hungarian Grand Prix"
)

if [[ $# -gt 0 ]]; then
  RACES=("$@")
fi

cd "$(dirname "$0")/.."

# Pin the interpreter. A bare `python` picks up whatever venv happens to be
# active — in this repo that is usually the *root* .venv (Streamlit deps), which
# has no sqlalchemy, so every ETL call dies at import. Prefer the backend venv,
# and fall back to PATH only where deps are installed system-wide (containers).
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x .venv/bin/python ]]; then
    PYTHON=".venv/bin/python"
  else
    PYTHON="python"
  fi
fi

# Fail fast on a broken environment. Without this the loop below turns an import
# error into one misleading "skipped" line per race and only surfaces at the
# final refresh-stats, pointing at the wrong step.
if ! "$PYTHON" -m app.etl --help >/dev/null 2>&1; then
  echo "error: '$PYTHON -m app.etl' is not runnable from $(pwd)." >&2
  echo "       Install the backend deps first:" >&2
  echo "         python -m venv .venv && .venv/bin/pip install -e \".[dev]\"" >&2
  echo "       (or set PYTHON=/path/to/python to choose the interpreter)." >&2
  "$PYTHON" -m app.etl --help >&2 || true
  exit 1
fi

echo "==> using interpreter: $PYTHON"

log="$(mktemp)"
trap 'rm -f "$log"' EXIT

failed=()
first=1

for gp in "${RACES[@]}"; do
  # Pace the requests — between races only, never before the first or after the
  # last, so hydrating a single race stays instant.
  if (( first )); then
    first=0
  elif (( DELAY > 0 )); then
    echo "    (pausing ${DELAY}s to stay under OpenF1 rate limits)"
    sleep "$DELAY"
  fi

  echo "==> hydrating $YEAR — $gp (R)"
  # hydrate exits 0 on success and non-zero only on an uncaught exception, so
  # the cause (no such session / network / DB) is in the traceback, not the
  # exit code — don't guess at it, print it.
  if ! "$PYTHON" -m app.etl hydrate --year "$YEAR" --gp "$gp" --session R 2>&1 | tee "$log"; then
    rc="${PIPESTATUS[0]}"
    echo "    FAILED: $gp (exit $rc) — $(tail -n 1 "$log")"
    failed+=("$gp")
    continue
  fi
done

echo "==> recomputing driver_stats for $YEAR"
"$PYTHON" -m app.etl refresh-stats --season "$YEAR"

if (( ${#failed[@]} > 0 )); then
  echo "done, with ${#failed[@]} of ${#RACES[@]} races failed:" >&2
  printf '  - %s\n' "${failed[@]}" >&2
  exit 1
fi

echo "done."
