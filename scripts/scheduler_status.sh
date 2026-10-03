#!/bin/sh
set -u
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
echo "Schedule: daily at 06:00 Europe/Berlin"
if launchctl print "gui/$(id -u)/com.quietcapital.prospective" 2>/dev/null; then :; else echo "Scheduler is not installed or not loaded."; fi
echo "Recent application log:"
if [ -f "$PROJECT_DIR/logs/prospective_daily.log" ]; then tail -n 20 "$PROJECT_DIR/logs/prospective_daily.log"; else echo "No daily run log yet."; fi
echo "Recent errors:"
if [ -f "$PROJECT_DIR/logs/prospective_daily.error.log" ]; then tail -n 20 "$PROJECT_DIR/logs/prospective_daily.error.log"; else echo "No error log yet."; fi
echo "Recent discovery scan:"
if [ -f "$PROJECT_DIR/logs/discovery_daily.log" ]; then tail -n 20 "$PROJECT_DIR/logs/discovery_daily.log"; else echo "No discovery run log yet."; fi
echo "Recent discovery errors:"
if [ -f "$PROJECT_DIR/logs/discovery_daily.error.log" ]; then tail -n 20 "$PROJECT_DIR/logs/discovery_daily.error.log"; else echo "No discovery error log yet."; fi
