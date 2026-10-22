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
if [[ "$EUID" -ne 0 ]]; then
    echo "Run with sudo so files can be installed with restricted ownership." >&2
    exit 1
fi
if [[ "$INSTALL_ROOT" != /* || "$RUNTIME_DIR" != /* ]]; then
    echo "Install and runtime directories must be absolute paths." >&2
    exit 1
fi
INSTALL_ROOT="$(realpath -m "$INSTALL_ROOT")"
RUNTIME_DIR="$(realpath -m "$RUNTIME_DIR")"
if [[ "$INSTALL_ROOT" == "/" || "$RUNTIME_DIR" == "/" \
    || "$RUNTIME_DIR" == "$INSTALL_ROOT" \
    || "$RUNTIME_DIR" == "$INSTALL_ROOT/"* ]]; then
    echo "Runtime directory must be outside the install root and cannot be /." >&2
    exit 1
fi
if [[ -z "$SERVICE_USER" ]] || ! id "$SERVICE_USER" >/dev/null 2>&1; then
    echo "Set WAVE_SERVICE_USER to the existing service account." >&2
    exit 1
fi
if ! getent group "$SERVICE_GROUP" >/dev/null 2>&1; then
    echo "Service group does not exist: $SERVICE_GROUP" >&2
    exit 1
fi
if [[ ! -f "$ARCHIVE" || ! -f "$ARCHIVE.sha256" ]]; then
    echo "Release archive or its .sha256 file is missing." >&2
    exit 1
fi
if [[ "$(uname -s)" != Linux ]]; then
    echo "Release installation is only supported on Linux." >&2
    exit 1
fi

ARCHIVE="$(realpath "$ARCHIVE")"
(cd "$(dirname "$ARCHIVE")" && sha256sum --check "$(basename "$ARCHIVE").sha256")

RELEASE_DIR="$INSTALL_ROOT/releases/wave-vms-$VERSION"
if [[ -e "$RELEASE_DIR" ]]; then
    echo "Versioned release already exists; refusing to overwrite: $RELEASE_DIR" >&2
    exit 1
fi
if tar -tzf "$ARCHIVE" | grep -Eq '(^/|(^|/)\.\.(/|$))'; then
    echo "Unsafe path detected in release archive." >&2
    exit 1
fi

mkdir -p "$INSTALL_ROOT/releases"
TEMP_DIR="$(mktemp -d "$INSTALL_ROOT/.staging.XXXXXX")"
trap 'rm -rf -- "$TEMP_DIR"' EXIT
tar -xzf "$ARCHIVE" -C "$TEMP_DIR"
SOURCE_DIR="$TEMP_DIR/wave-vms-$VERSION"
if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "Expected archive directory wave-vms-$VERSION was not found." >&2
    exit 1
fi
"$SOURCE_DIR/scripts/verify_release.sh" "$SOURCE_DIR"

mv "$SOURCE_DIR" "$RELEASE_DIR"
chown -R root:"$SERVICE_GROUP" "$RELEASE_DIR"
chmod -R u=rwX,g=rX,o=rX "$RELEASE_DIR"

for directory in \
    "$RUNTIME_DIR" \
    "$RUNTIME_DIR/config" \
    "$RUNTIME_DIR/models" \
    "$RUNTIME_DIR/known_faces" \
    "$RUNTIME_DIR/alert_images"; do
    if [[ ! -d "$directory" ]]; then
        install -d -o "$SERVICE_USER" -g "$SERVICE_GROUP" -m 0750 "$directory"
    fi
done

if [[ ! -e "$RUNTIME_DIR/.env" ]]; then
    install -o "$SERVICE_USER" -g "$SERVICE_GROUP" -m 0600 \
        "$RELEASE_DIR/.env.example" "$RUNTIME_DIR/.env"
    echo "Created placeholder $RUNTIME_DIR/.env; edit it before starting the service." >&2
fi
if [[ ! -e "$RUNTIME_DIR/config/cameras.json" ]]; then
    install -o "$SERVICE_USER" -g "$SERVICE_GROUP" -m 0640 \
        "$RELEASE_DIR/config/cameras.example.json" "$RUNTIME_DIR/config/cameras.json"
    echo "Created disabled camera template at $RUNTIME_DIR/config/cameras.json." >&2
fi
for model in "$RELEASE_DIR"/models/*.pt; do
    [[ -f "$model" ]] || continue
    target="$RUNTIME_DIR/models/$(basename "$model")"
    if [[ ! -e "$target" ]]; then
        install -o "$SERVICE_USER" -g "$SERVICE_GROUP" -m 0640 "$model" "$target"
    fi
done

for directory in "$RUNTIME_DIR" "$RUNTIME_DIR/known_faces" "$RUNTIME_DIR/alert_images" "$RUNTIME_DIR/models"; do
    if ! runuser -u "$SERVICE_USER" -- test -w "$directory"; then
        echo "Service account cannot write to runtime directory: $directory" >&2
        exit 1
    fi
done
for file in "$RUNTIME_DIR/.env" "$RUNTIME_DIR/config/cameras.json"; do
    if ! runuser -u "$SERVICE_USER" -- test -r "$file"; then
        echo "Service account cannot read runtime configuration: $file" >&2
        exit 1
    fi
done

echo "Staged release: $RELEASE_DIR"
echo "Runtime data:   $RUNTIME_DIR"
echo "No existing release, current symlink, service, database, or runtime file was overwritten."
echo "Edit external configuration and provision any missing default model assets before staging tests."
