#!/usr/bin/env bash
set -euo pipefail

INSTALL_ROOT="${WAVE_INSTALL_ROOT:-/opt/wave-vms}"
SERVICE_NAME="${WAVE_SERVICE_NAME:-wave-vms.service}"
HEALTH_URL="${WAVE_HEALTH_URL:-http://127.0.0.1:5000/api/cameras}"
CURRENT="$INSTALL_ROOT/current"
PREVIOUS="$INSTALL_ROOT/previous"

if [[ "$EUID" -ne 0 ]]; then
    echo "Run with sudo to switch releases and restart systemd." >&2
    exit 1
fi
if [[ ! -L "$PREVIOUS" || ! -d "$(readlink -f "$PREVIOUS")" ]]; then
    echo "No previous release pointer is available at $PREVIOUS." >&2
    exit 1
fi

current_target=""
if [[ -L "$CURRENT" ]]; then
    current_target="$(readlink -f "$CURRENT")"
fi
previous_target="$(readlink -f "$PREVIOUS")"
ln -sfn "$previous_target" "$CURRENT.rollback"
mv -Tf "$CURRENT.rollback" "$CURRENT"
if ! systemctl restart "$SERVICE_NAME"; then
    echo "Rollback service restart failed; inspect systemctl status and journal." >&2
    exit 1
fi
"$(dirname "$0")/health_check.sh" "$HEALTH_URL"
if [[ -n "$current_target" ]]; then
    ln -sfn "$current_target" "$PREVIOUS.new"
    mv -Tf "$PREVIOUS.new" "$PREVIOUS"
fi
echo "Rollback completed to $previous_target."
