#!/usr/bin/env bash
set -euo pipefail

ARCHIVE="${1:-}"
VERSION="${2:-}"
RUNTIME_DIR="${3:-/var/lib/wave-vms}"
SERVICE_USER="${WAVE_SERVICE_USER:-}"
SERVICE_GROUP="${WAVE_SERVICE_GROUP:-$SERVICE_USER}"
INSTALL_ROOT="${WAVE_INSTALL_ROOT:-/opt/wave-vms}"

if [[ -z "$ARCHIVE" || -z "$VERSION" || ! "$VERSION" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Usage: sudo WAVE_SERVICE_USER=<existing-account> $0 <release.tar.gz> <version> [runtime-directory]" >&2
    exit 2
fi
[[ "$EUID" -eq 0 ]] || { echo "Run with sudo." >&2; exit 1; }
[[ "$INSTALL_ROOT" == /* && "$RUNTIME_DIR" == /* ]] || {
    echo "Install and runtime directories must be absolute." >&2
    exit 1
}
INSTALL_ROOT="$(realpath -m "$INSTALL_ROOT")"
RUNTIME_DIR="$(realpath -m "$RUNTIME_DIR")"
if [[ "$INSTALL_ROOT" == / || "$RUNTIME_DIR" == / \
    || "$RUNTIME_DIR" == "$INSTALL_ROOT" || "$RUNTIME_DIR" == "$INSTALL_ROOT/"* ]]; then
    echo "Runtime directory must be outside the install root and cannot be /." >&2
    exit 1
fi
[[ -n "$SERVICE_USER" ]] && id "$SERVICE_USER" >/dev/null 2>&1 || {
    echo "Set WAVE_SERVICE_USER to the existing service account." >&2
    exit 1
}
getent group "$SERVICE_GROUP" >/dev/null 2>&1 || {
    echo "Service group does not exist: $SERVICE_GROUP" >&2
    exit 1
}
[[ -f "$ARCHIVE" && -f "$ARCHIVE.sha256" ]] || {
    echo "Archive and its .sha256 sidecar are required." >&2
    exit 1
}
[[ "$(uname -s)" == Linux ]] || { echo "Installation requires Linux." >&2; exit 1; }

ARCHIVE="$(realpath "$ARCHIVE")"
(cd "$(dirname "$ARCHIVE")" && sha256sum --check "$(basename "$ARCHIVE").sha256")
RELEASE_DIR="$INSTALL_ROOT/releases/wave-vms-$VERSION"
[[ ! -e "$RELEASE_DIR" ]] || {
    echo "Refusing to overwrite existing version: $RELEASE_DIR" >&2
    exit 1
}
if tar -tzf "$ARCHIVE" | grep -Eq '(^/|(^|/)\.\.(/|$))'; then
    echo "Unsafe path detected in release archive." >&2
    exit 1
fi

mkdir -p "$INSTALL_ROOT/releases"
TEMP_DIR="$(mktemp -d "$INSTALL_ROOT/.staging.XXXXXX")"
trap 'rm -rf -- "$TEMP_DIR"' EXIT
tar -xzf "$ARCHIVE" -C "$TEMP_DIR"
STAGED_RELEASE="$TEMP_DIR/wave-vms-$VERSION"
[[ -d "$STAGED_RELEASE" ]] || {
    echo "Archive did not contain wave-vms-$VERSION." >&2
    exit 1
}
"$STAGED_RELEASE/scripts/verify_release.sh" "$STAGED_RELEASE"
mv "$STAGED_RELEASE" "$RELEASE_DIR"
chown -R root:"$SERVICE_GROUP" "$RELEASE_DIR"
chmod -R u=rwX,g=rX,o=rX "$RELEASE_DIR"

for directory in "$RUNTIME_DIR" "$RUNTIME_DIR/config" "$RUNTIME_DIR/models" \
    "$RUNTIME_DIR/known_faces" "$RUNTIME_DIR/alert_images"; do
    if [[ ! -d "$directory" ]]; then
        install -d -o "$SERVICE_USER" -g "$SERVICE_GROUP" -m 0750 "$directory"
    fi
done
if [[ ! -e "$RUNTIME_DIR/.env" ]]; then
    install -o "$SERVICE_USER" -g "$SERVICE_GROUP" -m 0600 \
        "$RELEASE_DIR/.env.example" "$RUNTIME_DIR/.env"
    echo "Created placeholder $RUNTIME_DIR/.env; configure it before service startup." >&2
fi
if [[ ! -e "$RUNTIME_DIR/config/cameras.json" ]]; then
    install -o "$SERVICE_USER" -g "$SERVICE_GROUP" -m 0640 \
        "$RELEASE_DIR/config/cameras.example.json" "$RUNTIME_DIR/config/cameras.json"
    echo "Created an empty camera configuration; preserve or provision the intended configuration." >&2
fi
for model in "$RELEASE_DIR"/models/*.pt; do
    [[ -f "$model" ]] || continue
    target="$RUNTIME_DIR/models/$(basename "$model")"
    if [[ ! -e "$target" ]]; then
        install -o "$SERVICE_USER" -g "$SERVICE_GROUP" -m 0640 "$model" "$target"
    fi
done
for directory in "$RUNTIME_DIR" "$RUNTIME_DIR/known_faces" "$RUNTIME_DIR/alert_images" "$RUNTIME_DIR/models"; do
    runuser -u "$SERVICE_USER" -- test -w "$directory" || {
        echo "Service account cannot write to $directory" >&2
        exit 1
    }
done
for file in "$RUNTIME_DIR/.env" "$RUNTIME_DIR/config/cameras.json"; do
    runuser -u "$SERVICE_USER" -- test -r "$file" || {
        echo "Service account cannot read $file" >&2
        exit 1
    }
done

echo "Installed versioned release: $RELEASE_DIR"
echo "External runtime directory:  $RUNTIME_DIR"
echo "No service, current symlink, database, or existing runtime file was changed."
echo "Review the service template and perform staging checks before switching production."
