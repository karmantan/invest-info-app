#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TARGET="$HOME/Library/LaunchAgents/com.quietcapital.prospective.plist"
mkdir -p "$HOME/Library/LaunchAgents" "$PROJECT_DIR/logs"
sed "s|__PROJECT_DIR__|$PROJECT_DIR|g" "$PROJECT_DIR/scripts/com.quietcapital.prospective.plist.template" > "$TARGET"
plutil -lint "$TARGET"
launchctl bootout "gui/$(id -u)" "$TARGET" >/dev/null 2>&1 || true
launchctl bootstrap "gui/$(id -u)" "$TARGET"
echo "Installed daily 06:00 Europe/Berlin paper scanner: $TARGET"
