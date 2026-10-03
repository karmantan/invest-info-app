#!/bin/sh
set -u
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_DIR" || exit 1
mkdir -p logs
"$PROJECT_DIR/.venv/bin/python" scripts/spreads.py collect >>logs/spreads.log 2>>logs/spreads.error.log
