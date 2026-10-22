#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-}"
INSTALL_ROOT="${WAVE_INSTALL_ROOT:-/opt/wave-vms}"
SERVICE_NAME="${WAVE_SERVICE_NAME:-wave-vms.service}"
HEALTH_URL="${WAVE_HEALTH_URL:-http://127.0.0.1:5000/api/cameras}"
RELEASE_DIR="$INSTALL_ROOT/releases/wave-vms-$VERSION"
CURRENT="$INSTALL_ROOT/current"
PREVIOUS="$INSTALL_ROOT/previous"

if [[ -z "$VERSION" || ! "$VERSION" =~ ^[A-Za-z0-9._-]+$ || ! -x "$RELEASE_DIR/wave-vms" ]]; then
    echo "Usage: sudo $0 <installed-version>" >&2
    exit 2
fi
if [[ "$EUID" -ne 0 ]]; then
    echo "Run with sudo to update the release pointer and restart systemd." >&2
    exit 1
fi
exec_start="$(systemctl show --property=ExecStart --value "$SERVICE_NAME")"
if [[ "$exec_start" != *"$CURRENT/wave-vms"* ]]; then
    echo "$SERVICE_NAME must execute $CURRENT/wave-vms before activation." >&2
    exit 1
fi
"$RELEASE_DIR/scripts/verify_release.sh" "$RELEASE_DIR"
"$(dirname "$0")/health_check.sh" "$HEALTH_URL" >/dev/null

old_target=""
if [[ -L "$CURRENT" ]]; then
    old_target="$(readlink -f "$CURRENT")"
    ln -sfn "$old_target" "$PREVIOUS.new"
    mv -Tf "$PREVIOUS.new" "$PREVIOUS"
fi
ln -sfn "$RELEASE_DIR" "$CURRENT.new"
mv -Tf "$CURRENT.new" "$CURRENT"

rollback_pointer() {
    if [[ -n "$old_target" ]]; then
        ln -sfn "$old_target" "$CURRENT.rollback"
        mv -Tf "$CURRENT.rollback" "$CURRENT"
    else
        rm -f "$CURRENT"
    fi
    systemctl restart "$SERVICE_NAME" || true
}

if ! systemctl restart "$SERVICE_NAME"; then
    rollback_pointer
    echo "Service restart failed; restored the previous release pointer." >&2
    exit 1
fi

for attempt in $(seq 1 12); do
    if "$(dirname "$0")/health_check.sh" "$HEALTH_URL" >/dev/null 2>&1; then
        echo "Activated wave-vms-$VERSION and passed the HTTP health check."
        exit 0
    fi
    sleep 5
done

rollback_pointer
echo "New release failed its health check; restored the prior release pointer." >&2
exit 1
