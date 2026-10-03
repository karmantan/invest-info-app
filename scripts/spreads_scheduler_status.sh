#!/bin/sh
set -u
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
echo "Schedule: every 15 minutes; collector gates to Xetra sessions and 09:00–17:30 Europe/Berlin"
launchctl print "gui/$(id -u)/com.quietcapital.spreads" 2>/dev/null || echo "Spread scheduler is not installed or not loaded."
"$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/spreads.py" status
tail -n 20 "$PROJECT_DIR/logs/spreads.log" 2>/dev/null || true
tail -n 20 "$PROJECT_DIR/logs/spreads.error.log" 2>/dev/null || true
