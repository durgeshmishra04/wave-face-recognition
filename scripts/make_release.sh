#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${1:-}"
OUT_ROOT="${2:-$ROOT/release}"
DIST_DIR="$ROOT/dist/wave-vms"

if [[ ! "$VERSION" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Usage: $0 <release-version> [output-directory]" >&2
    exit 2
fi
if [[ ! -x "$DIST_DIR/wave-vms" || ! -d "$DIST_DIR/_internal" ]]; then
    echo "Linux onedir output not found at $DIST_DIR; run build_linux.sh first." >&2
    exit 1
fi
if [[ ! -f "$ROOT/build/linux-build-info.txt" || ! -f "$ROOT/build/linux-dependencies.lock" ]]; then
    echo "Linux build metadata is missing; run build_linux.sh first." >&2
    exit 1
fi
mkdir -p "$OUT_ROOT"
OUT_ROOT="$(cd "$OUT_ROOT" && pwd)"
if [[ -e "$OUT_ROOT/wave-vms-$VERSION" \
    || -e "$OUT_ROOT/wave-vms-$VERSION.tar.gz" \
    || -e "$OUT_ROOT/wave-vms-$VERSION-install.sh" \
    || -e "$OUT_ROOT/wave-vms-$VERSION-install.sh.sha256" ]]; then
    echo "Release output already exists; choose a new version or output directory." >&2
    exit 1
fi

STAGE="$OUT_ROOT/wave-vms-$VERSION"
mkdir -p "$STAGE/config" "$STAGE/models" "$STAGE/scripts"
cp -a "$DIST_DIR/." "$STAGE/"
cp "$ROOT/.env.example" "$STAGE/.env.example"
cp "$ROOT/README_DEPLOYMENT.md" "$STAGE/README_DEPLOYMENT.md"
cp "$ROOT/packaging/wave-vms.service" "$STAGE/wave-vms.service"
cp "$ROOT/packaging/wave-vms-build.service" "$STAGE/wave-vms-build.service"
cp "$ROOT/scripts/health_check.sh" \
    "$ROOT/scripts/verify_release.sh" \
    "$ROOT/scripts/install_release.sh" \
    "$ROOT/scripts/activate_release.sh" \
    "$ROOT/scripts/rollback.sh" \
    "$ROOT/scripts/switch_mode.sh" \
    "$STAGE/scripts/"
chmod 0755 "$STAGE"/scripts/*.sh
cp "$ROOT/scripts/install_release.sh" "$OUT_ROOT/wave-vms-$VERSION-install.sh"
chmod 0755 "$OUT_ROOT/wave-vms-$VERSION-install.sh"
(
    cd "$OUT_ROOT"
    sha256sum "wave-vms-$VERSION-install.sh" \
        > "wave-vms-$VERSION-install.sh.sha256"
)
cp "$ROOT/build/linux-build-info.txt" "$STAGE/BUILD_INFO.txt"
cp "$ROOT/build/linux-dependencies.lock" "$STAGE/DEPENDENCIES.lock"

python3.11 - "$ROOT/config/cameras.json" "$STAGE/config/cameras.example.json" <<'PY'
import json
import sys

source, destination = sys.argv[1:]
with open(source, encoding="utf-8") as camera_file:
    cameras = json.load(camera_file)
for index, camera in enumerate(cameras, start=1):
    camera["name"] = f"Camera {index:03d}"
    camera["ip"] = ""
    camera["nvr_ip"] = ""
    camera["enabled"] = False
with open(destination, "w", encoding="utf-8") as camera_file:
    json.dump(cameras, camera_file, indent=2)
    camera_file.write("\n")
PY

asset_list="$STAGE/models/ASSET_MANIFEST.txt"
{
    echo "Model assets copied from the repository; no weights are synthesized."
    for model in yolo11n.pt yolo26l.pt yolo11m-pose.pt yolo26n_ppe.pt salahyolo26.pt barrel.pt; do
        if [[ -f "$ROOT/$model" ]]; then
            cp "$ROOT/$model" "$STAGE/models/$model"
            printf 'PRESENT %s\n' "$model"
        else
            printf 'MISSING %s\n' "$model"
        fi
    done
    echo "InsightFace buffalo_l is not in the repository; provision its model cache separately."
} > "$asset_list"

(cd "$STAGE" && find . -type f ! -name SHA256SUMS -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum > SHA256SUMS)

ARCHIVE="$OUT_ROOT/wave-vms-$VERSION.tar.gz"
tar -C "$OUT_ROOT" -czf "$ARCHIVE" "wave-vms-$VERSION"
(cd "$OUT_ROOT" && sha256sum "$(basename "$ARCHIVE")" > "$(basename "$ARCHIVE").sha256")

echo "Release directory: $STAGE"
echo "Release archive:   $ARCHIVE"
echo "Archive checksum:  $ARCHIVE.sha256"
echo "Model asset report: $asset_list"
if grep -q '^MISSING ' "$asset_list"; then
    echo "WARNING: one or more default model weights are missing; provision them externally before model testing." >&2
fi
