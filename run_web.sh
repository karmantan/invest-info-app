#!/bin/sh
# Start the website locally (the hosted version runs the same app on Render).
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"
PYTHON="$SCRIPT_DIR/.venv/bin/python"; [ -x "$PYTHON" ] || PYTHON=python3
exec "$PYTHON" -m streamlit run webapp/app.py
