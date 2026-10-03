#!/bin/sh
set -eu
TARGET="$HOME/Library/LaunchAgents/com.quietcapital.prospective.plist"
launchctl bootout "gui/$(id -u)" "$TARGET" >/dev/null 2>&1 || true
if [ -f "$TARGET" ]; then rm "$TARGET"; fi
echo "Removed com.quietcapital.prospective. Paper records and logs were preserved."
