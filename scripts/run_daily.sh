#!/bin/sh
set -u
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_DIR" || exit 1
mkdir -p logs
# LaunchAgent supplies only /usr/bin:/bin:/usr/sbin:/sbin.  Keep imports rooted
# at this checkout so the ingestion and prospective stages run consistently.
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
if [ -f .env ]; then set -a; . "$PROJECT_DIR/.env"; set +a; fi
PYTHON_BIN="${DAILY_PYTHON:-$PROJECT_DIR/.venv/bin/python}"
LOG_DIR="${DAILY_LOG_DIR:-$PROJECT_DIR/logs}"
mkdir -p "$LOG_DIR"
INGESTION_FAILED=""
OVERALL_STATUS=0
if ! "$PYTHON_BIN" scripts/manage.py ingest-real --refresh >>"$LOG_DIR/prospective_daily.log" 2>>"$LOG_DIR/prospective_daily.error.log"; then
  INGESTION_FAILED="market, rates, VIX, or FX refresh failed"
  OVERALL_STATUS=1
  printf '%s DAILY FAILURE: ingestion step failed\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$LOG_DIR/prospective_daily.error.log"
fi
if [ -n "$INGESTION_FAILED" ]; then
  "$PYTHON_BIN" scripts/prospective.py daily --ingestion-failed "$INGESTION_FAILED" >>"$LOG_DIR/prospective_daily.log" 2>>"$LOG_DIR/prospective_daily.error.log" || {
    OVERALL_STATUS=1
    printf '%s DAILY FAILURE: prospective step failed after ingestion failure\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$LOG_DIR/prospective_daily.error.log"
  }
else
  "$PYTHON_BIN" scripts/prospective.py daily >>"$LOG_DIR/prospective_daily.log" 2>>"$LOG_DIR/prospective_daily.error.log" || {
    OVERALL_STATUS=1
    printf '%s DAILY FAILURE: prospective step failed\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$LOG_DIR/prospective_daily.error.log"
  }
fi
# Discovery is isolated from the frozen prospective job: a discovery failure
# is logged but can never change the validation experiment's exit status/data.
"$PYTHON_BIN" scripts/discovery.py daily --refresh >>"$LOG_DIR/discovery_daily.log" 2>>"$LOG_DIR/discovery_daily.error.log" || true
exit "$OVERALL_STATUS"
