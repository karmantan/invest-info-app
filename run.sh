#!/bin/sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export INVEST_DASHBOARD_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%S%z)"
exec "$SCRIPT_DIR/.venv/bin/python" -m streamlit run "$SCRIPT_DIR/dashboard/app.py" --server.headless=true --browser.gatherUsageStats=false
