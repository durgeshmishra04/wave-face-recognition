#!/usr/bin/env bash
set -euo pipefail

RELEASE_DIR="${1:-}"
if [[ -z "$RELEASE_DIR" || ! -d "$RELEASE_DIR" ]]; then
    echo "Usage: $0 <release-directory>" >&2
    exit 2
fi
RELEASE_DIR="$(cd "$RELEASE_DIR" && pwd)"

if [[ "$(uname -s)" != Linux ]]; then
    echo "Release verification must run on Linux." >&2
    exit 1
fi
for required in wave-vms _internal SHA256SUMS config/cameras.example.json .env.example models/ASSET_MANIFEST.txt BUILD_INFO.txt DEPENDENCIES.lock; do
    if [[ ! -e "$RELEASE_DIR/$required" ]]; then
        echo "Required release item is missing: $required" >&2
        exit 1
    fi
done
if [[ ! -x "$RELEASE_DIR/wave-vms" ]]; then
    echo "Executable bit is missing from wave-vms." >&2
    exit 1
fi
if find "$RELEASE_DIR" -maxdepth 1 -type f \( -name '*.py' -o -name 'app.py' -o -name 'alcon_stream_new.py' \) | grep -q .; then
    echo "Python source files must not be copied to the runtime release." >&2
    exit 1
fi

(
    cd "$RELEASE_DIR"
    sha256sum --check --strict SHA256SUMS
)

if command -v file >/dev/null 2>&1 && ! file "$RELEASE_DIR/wave-vms" | grep -q 'ELF'; then
    echo "Release executable is not a Linux ELF binary." >&2
    exit 1
fi
source /etc/os-release
expected_arch="$(sed -n 's/^BUILD_ARCH=//p' "$RELEASE_DIR/BUILD_INFO.txt")"
expected_ubuntu="$(sed -n 's/^BUILD_UBUNTU_VERSION=//p' "$RELEASE_DIR/BUILD_INFO.txt")"
expected_glibc="$(sed -n 's/^BUILD_GLIBC=//p' "$RELEASE_DIR/BUILD_INFO.txt")"
if [[ "${ID:-}" != ubuntu || "${VERSION_ID:-}" != "$expected_ubuntu" ]]; then
    echo "Release requires Ubuntu $expected_ubuntu; found ${PRETTY_NAME:-unknown}." >&2
    exit 1
fi
if [[ -n "$expected_arch" && "$(uname -m)" != "$expected_arch" ]]; then
    echo "Release architecture is $expected_arch, but this host is $(uname -m)." >&2
    exit 1
fi
if [[ -n "$expected_glibc" && "$(getconf GNU_LIBC_VERSION 2>/dev/null || echo unknown)" != "$expected_glibc" ]]; then
    echo "Release glibc is $expected_glibc; host has $(getconf GNU_LIBC_VERSION 2>/dev/null || echo unknown)." >&2
    exit 1
fi
if command -v ldd >/dev/null 2>&1; then
    if ldd "$RELEASE_DIR/wave-vms" 2>&1 | grep -q 'not found'; then
        echo "Unresolved shared library dependencies detected by ldd." >&2
        exit 1
    fi
fi

missing=0
for model in yolo11n.pt yolo26l.pt yolo11m-pose.pt; do
    if [[ ! -f "$RELEASE_DIR/models/$model" ]]; then
        echo "Missing default model asset: models/$model" >&2
        missing=1
    fi
done
if [[ "$missing" -ne 0 ]]; then
    echo "WARNING: default model assets are missing; this release is not ready for production." >&2
fi

echo "Structural release checks passed. This does not verify model loading, GPU, PostgreSQL, camera, notification, or API behavior."
