#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-}"
SERVICE_USER="${2:-}"
SERVICE_GROUP="${3:-}"
RUNTIME_DIR="${4:-}"
UNIT_NAME="${5:-wave-vms-build.service}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_ROOT="${WAVE_INSTALL_ROOT:-/opt/wave-vms}"

if [[ -z "$VERSION" || ! "$VERSION" =~ ^[A-Za-z0-9._-]+$ \
    || -z "$SERVICE_USER" || -z "$SERVICE_GROUP" || -z "$RUNTIME_DIR" ]]; then
    echo "Usage: sudo $0 <version> <existing-service-user> <existing-service-group> <absolute-runtime-dir> [unit-name.service]" >&2
    exit 2
fi
[[ "$EUID" -eq 0 ]] || { echo "Run with sudo." >&2; exit 1; }
[[ "$UNIT_NAME" =~ ^[A-Za-z0-9_.@-]+\.service$ ]] || {
    echo "Unit name must be a simple systemd .service name." >&2
    exit 1
}
id "$SERVICE_USER" >/dev/null 2>&1 || { echo "Unknown service user: $SERVICE_USER" >&2; exit 1; }
getent group "$SERVICE_GROUP" >/dev/null 2>&1 || { echo "Unknown service group: $SERVICE_GROUP" >&2; exit 1; }
[[ "$RUNTIME_DIR" == /* ]] || { echo "Runtime directory must be absolute." >&2; exit 1; }

RELEASE_DIR="$(realpath -m "$INSTALL_ROOT/releases/wave-vms-$VERSION")"
[[ -x "$RELEASE_DIR/wave-vms" ]] || {
    echo "Installed executable not found: $RELEASE_DIR/wave-vms" >&2
    exit 1
}
UNIT_PATH="/etc/systemd/system/$UNIT_NAME"
[[ ! -e "$UNIT_PATH" ]] || {
    echo "Refusing to overwrite existing unit: $UNIT_PATH" >&2
    exit 1
}
TEMP_UNIT="$(mktemp /etc/systemd/system/.wave-vms.XXXXXX)"
trap 'rm -f -- "$TEMP_UNIT"' EXIT

python3 - "$SCRIPT_DIR/wave-vms-build.service.example" "$TEMP_UNIT" \
    "$SERVICE_USER" "$SERVICE_GROUP" "$RUNTIME_DIR" "$RELEASE_DIR/wave-vms" <<'PY'
from pathlib import Path
import sys

template, destination, user, group, runtime, executable = sys.argv[1:]
text = Path(template).read_text(encoding="utf-8")
text = text.replace("REPLACE_WITH_EXISTING_SERVICE_USER", user)
text = text.replace("REPLACE_WITH_EXISTING_SERVICE_GROUP", group)
text = text.replace("/var/lib/wave-vms", runtime)
text = text.replace("/opt/wave-vms/releases/wave-vms-VERSION/wave-vms", executable)
Path(destination).write_text(text, encoding="utf-8")
PY
chmod 0644 "$TEMP_UNIT"
mv "$TEMP_UNIT" "$UNIT_PATH"
trap - EXIT
systemctl daemon-reload

echo "Configured unit: $UNIT_PATH"
echo "It was not enabled, started, or used to stop another service."
echo "The application reads credentials from $RUNTIME_DIR/.env."
