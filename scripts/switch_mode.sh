#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SOURCE_SERVICE="${WAVE_SOURCE_SERVICE:-wave-vms-source.service}"
BUILD_SERVICE="${WAVE_BUILD_SERVICE:-wave-vms-build.service}"
HEALTH_URL="${WAVE_HEALTH_URL:-http://127.0.0.1:5000/api/cameras}"

if [[ "$MODE" != source && "$MODE" != build ]]; then
    echo "Usage: sudo $0 <source|build>" >&2
    exit 2
fi
if [[ "$EUID" -ne 0 ]]; then
    echo "Run with sudo to switch systemd services." >&2
    exit 1
fi

if [[ "$MODE" == source ]]; then
    selected="$SOURCE_SERVICE"
    other="$BUILD_SERVICE"
else
    selected="$BUILD_SERVICE"
    other="$SOURCE_SERVICE"
fi

if systemctl is-active --quiet "$selected"; then
    if "$(dirname "$0")/health_check.sh" "$HEALTH_URL"; then
        echo "$selected is already active and healthy."
        exit 0
    fi
    echo "$selected is active but unhealthy; refusing to start another mode." >&2
    exit 1
fi

if systemctl is-active --quiet "$other"; then
    systemctl stop "$other"
fi
if systemctl is-active --quiet "$other"; then
    echo "Could not stop $other; refusing to start $selected." >&2
    exit 1
fi

systemctl start "$selected"
for attempt in $(seq 1 12); do
    if systemctl is-active --quiet "$selected" \
        && "$(dirname "$0")/health_check.sh" "$HEALTH_URL" >/dev/null 2>&1; then
        echo "Started $selected and verified $HEALTH_URL."
        exit 0
    fi
    sleep 5
done

systemctl stop "$selected" || true
echo "$selected did not pass its health check; it was stopped. Inspect: journalctl -u $selected -n 200 --no-pager" >&2
exit 1
