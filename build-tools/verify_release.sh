#!/usr/bin/env bash
set -euo pipefail

RELEASE_DIR="${1:-}"
if [[ -z "$RELEASE_DIR" || ! -d "$RELEASE_DIR" ]]; then
    echo "Usage: $0 <release-directory>" >&2
    exit 2
fi
RELEASE_DIR="$(cd "$RELEASE_DIR" && pwd)"
if [[ "$(uname -s)" != Linux ]]; then
    echo "Release verification requires Linux." >&2
    exit 1
fi
for required in wave-vms _internal SHA256SUMS config/cameras.example.json \
    .env.example models/ASSET_MANIFEST.txt BUILD_INFO.txt DEPENDENCIES.lock; do
    if [[ ! -e "$RELEASE_DIR/$required" ]]; then
        echo "Required release item is missing: $required" >&2
        exit 1
    fi
done
if [[ ! -x "$RELEASE_DIR/wave-vms" ]]; then
    echo "Executable bit is missing from wave-vms." >&2
    exit 1
fi
if find "$RELEASE_DIR" -type f \( -name '*.py' -o -name '*.pyc' \) -print -quit | grep -q .; then
    echo "Python source files must not be present in the release." >&2
    exit 1
fi
if [[ -e "$RELEASE_DIR/.env" || -e "$RELEASE_DIR/pythonai.json" ]]; then
    echo "Credentials must remain outside the release." >&2
    exit 1
fi
(cd "$RELEASE_DIR" && sha256sum --check --strict SHA256SUMS)
if command -v file >/dev/null 2>&1 && ! file "$RELEASE_DIR/wave-vms" | grep -q ELF; then
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
if [[ "$(uname -m)" != "$expected_arch" ]]; then
    echo "Release architecture $expected_arch does not match $(uname -m)." >&2
    exit 1
fi
if [[ "$(getconf GNU_LIBC_VERSION 2>/dev/null || echo unknown)" != "$expected_glibc" ]]; then
    echo "Release glibc $expected_glibc does not match this host." >&2
    exit 1
fi
if command -v ldd >/dev/null 2>&1; then
    while IFS= read -r -d '' native_file; do
        if ! dependency_report="$(ldd "$native_file" 2>&1)"; then
            echo "Could not inspect native dependencies for $native_file:" >&2
            echo "$dependency_report" >&2
            exit 1
        fi
        if grep -q 'not found' <<< "$dependency_report"; then
            echo "Unresolved shared-library dependencies in $native_file:" >&2
            echo "$dependency_report" >&2
            exit 1
        fi
    done < <(find "$RELEASE_DIR" -type f \
        \( -name wave-vms -o -name '*.so' -o -name '*.so.*' \) -print0)
fi
echo "Structural checks passed. Application APIs, models, GPU, database, cameras, and alerts still require staging tests."
