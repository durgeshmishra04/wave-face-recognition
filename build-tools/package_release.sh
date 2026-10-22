#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$(cd -- "$TOOLS_DIR/.." && pwd)"
OUTPUT_ROOT="${WAVE_BUILD_OUTPUT:-$(dirname "$SOURCE_ROOT")/wave-vms-compiled}"
BUILD_ROOT="$OUTPUT_ROOT/.build"
DIST_DIR="$BUILD_ROOT/dist/wave-vms"
VERSION="${1:-}"

if [[ ! "$VERSION" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Usage: $0 <release-version>" >&2
    exit 2
fi
for required in "$DIST_DIR/wave-vms" "$DIST_DIR/_internal" \
    "$BUILD_ROOT/linux-build-info.txt" "$BUILD_ROOT/linux-dependencies.lock"; do
    if [[ ! -e "$required" ]]; then
        echo "Required build output is missing: $required" >&2
        echo "Run build-tools/build_linux.sh <release-version> on a compatible Ubuntu host." >&2
        exit 1
    fi
done
if [[ "$(uname -s)" != Linux ]]; then
    echo "Packaging a Linux release must run on Linux." >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"
RELEASE_DIR="$OUTPUT_ROOT/wave-vms-$VERSION"
ARCHIVE="$OUTPUT_ROOT/wave-vms-$VERSION-ubuntu.tar.gz"
if [[ -e "$RELEASE_DIR" || -e "$ARCHIVE" || -e "$ARCHIVE.sha256" ]]; then
    echo "Release output already exists; choose a new version or output directory." >&2
    exit 1
fi

mkdir -p "$RELEASE_DIR/config" "$RELEASE_DIR/models" "$RELEASE_DIR/scripts"
cp -a "$DIST_DIR/." "$RELEASE_DIR/"
cp "$SOURCE_ROOT/.env.example" "$RELEASE_DIR/.env.example"
cp "$BUILD_ROOT/linux-build-info.txt" "$RELEASE_DIR/BUILD_INFO.txt"
cp "$BUILD_ROOT/linux-dependencies.lock" "$RELEASE_DIR/DEPENDENCIES.lock"
cp "$TOOLS_DIR/README.md" "$RELEASE_DIR/README_DEPLOYMENT.md"
cp "$TOOLS_DIR/wave-vms-build.service.example" "$RELEASE_DIR/wave-vms-build.service.example"
cp "$TOOLS_DIR/verify_release.sh" "$TOOLS_DIR/health_check.sh" \
    "$TOOLS_DIR/install_release.sh" "$TOOLS_DIR/configure_service.sh" \
    "$RELEASE_DIR/scripts/"
chmod 0755 "$RELEASE_DIR/wave-vms" "$RELEASE_DIR/scripts/"*.sh

printf '[]\n' > "$RELEASE_DIR/config/cameras.example.json"
{
    echo "Model assets included when present in the source checkout."
    echo "Persistent production data and .insightface/buffalo_l are not included."
    for model in yolo11n.pt yolo26l.pt yolo11m-pose.pt yolo26n_ppe.pt salahyolo26.pt barrel.pt; do
        if [[ -f "$SOURCE_ROOT/$model" ]]; then
            cp "$SOURCE_ROOT/$model" "$RELEASE_DIR/models/$model"
            printf 'PRESENT %s\n' "$model"
        else
            printf 'MISSING %s\n' "$model"
        fi
    done
} > "$RELEASE_DIR/models/ASSET_MANIFEST.txt"

if find "$RELEASE_DIR" -type f \( -name '*.py' -o -name '*.pyc' \) -print -quit | grep -q .; then
    echo "Source Python files unexpectedly entered the release; refusing to package." >&2
    exit 1
fi
if [[ -e "$SOURCE_ROOT/.env" || -e "$SOURCE_ROOT/pythonai.json" ]]; then
    echo "Local credentials detected in source root; they are intentionally excluded." >&2
fi

(
    cd "$RELEASE_DIR"
    find . -type f ! -name SHA256SUMS -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 sha256sum > SHA256SUMS
)
tar -C "$OUTPUT_ROOT" -czf "$ARCHIVE" "$(basename "$RELEASE_DIR")"
(cd "$OUTPUT_ROOT" && sha256sum "$(basename "$ARCHIVE")" > "$(basename "$ARCHIVE").sha256")

echo "Release directory: $RELEASE_DIR"
echo "Release archive:   $ARCHIVE"
echo "Checksum:          $ARCHIVE.sha256"
echo "Build output:      $BUILD_ROOT"
