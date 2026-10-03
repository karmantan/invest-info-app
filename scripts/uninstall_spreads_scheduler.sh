#!/bin/sh
set -eu
TARGET="$HOME/Library/LaunchAgents/com.quietcapital.spreads.plist"
launchctl bootout "gui/$(id -u)" "$TARGET" >/dev/null 2>&1 || true
if [ -f "$TARGET" ]; then rm "$TARGET"; fi
echo "Removed com.quietcapital.spreads. Observations, attempts, cooldown and logs were preserved."
