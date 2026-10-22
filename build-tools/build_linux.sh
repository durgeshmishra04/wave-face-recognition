#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$(cd -- "$TOOLS_DIR/.." && pwd)"
OUTPUT_ROOT="${WAVE_BUILD_OUTPUT:-$(dirname "$SOURCE_ROOT")/wave-vms-compiled}"
BUILD_ROOT="$OUTPUT_ROOT/.build"
BUILD_VENV="${BUILD_VENV:-$OUTPUT_ROOT/.build-venv}"
VERSION="${1:-}"

if [[ ! "$VERSION" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Usage: $0 <release-version>" >&2
    exit 2
fi
if [[ "$(uname -s)" != Linux ]]; then
    echo "A Linux host is required; this script will not create a Windows executable." >&2
    exit 1
fi
if [[ -z "${TARGET_UBUNTU_RELEASE:-}" || -z "${TARGET_ARCH:-}" || -z "${TARGET_GLIBC_VERSION:-}" ]]; then
    echo "Set TARGET_UBUNTU_RELEASE, TARGET_ARCH, and TARGET_GLIBC_VERSION from the target VM." >&2
    exit 1
fi
if [[ ! -r /etc/os-release ]]; then
    echo "Cannot determine the Linux distribution from /etc/os-release." >&2
    exit 1
fi

source /etc/os-release
if [[ "${ID:-}" != ubuntu || "${VERSION_ID:-}" != "$TARGET_UBUNTU_RELEASE" ]]; then
    echo "Build host must be Ubuntu $TARGET_UBUNTU_RELEASE; found ${PRETTY_NAME:-unknown}." >&2
    exit 1
fi
if [[ "$(uname -m)" != "$TARGET_ARCH" ]]; then
    echo "Build architecture $(uname -m) does not match target $TARGET_ARCH." >&2
    exit 1
fi
actual_glibc="$(getconf GNU_LIBC_VERSION 2>/dev/null || echo unknown)"
if [[ "$actual_glibc" != "$TARGET_GLIBC_VERSION" ]]; then
    echo "Build glibc $actual_glibc does not match target $TARGET_GLIBC_VERSION." >&2
    exit 1
fi
if ! command -v python3.11 >/dev/null 2>&1; then
    echo "Python 3.11 is required; install it on this compatible Ubuntu host." >&2
    exit 1
fi
python3.11 -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
if [[ -n "${PIP_CONSTRAINT_FILE:-}" ]]; then
    if [[ ! -f "$PIP_CONSTRAINT_FILE" ]]; then
        echo "PIP_CONSTRAINT_FILE does not exist: $PIP_CONSTRAINT_FILE" >&2
        exit 1
    fi
    export PIP_CONSTRAINT="$(realpath "$PIP_CONSTRAINT_FILE")"
fi

mkdir -p "$BUILD_ROOT"
export WAVE_SOURCE_ROOT="$SOURCE_ROOT"
python3.11 -m venv "$BUILD_VENV"
"$BUILD_VENV/bin/python" -m pip install --upgrade pip
"$BUILD_VENV/bin/python" -m pip install \
    -r "$SOURCE_ROOT/requirements.txt" \
    -r "$SOURCE_ROOT/requirements-build.txt"
"$BUILD_VENV/bin/python" -m PyInstaller \
    --clean \
    --noconfirm \
    --distpath "$BUILD_ROOT/dist" \
    --workpath "$BUILD_ROOT/work" \
    "$TOOLS_DIR/wave-vms.spec"

EXECUTABLE="$BUILD_ROOT/dist/wave-vms/wave-vms"
if [[ ! -x "$EXECUTABLE" || ! -d "$BUILD_ROOT/dist/wave-vms/_internal" ]]; then
    echo "PyInstaller did not produce a complete onedir application." >&2
    exit 1
fi
if command -v file >/dev/null 2>&1 && ! file "$EXECUTABLE" | grep -q 'ELF'; then
    echo "PyInstaller output is not a Linux ELF executable." >&2
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
    done < <(find "$BUILD_ROOT/dist/wave-vms" -type f \
        \( -name wave-vms -o -name '*.so' -o -name '*.so.*' \) -print0)
fi

"$BUILD_VENV/bin/python" -m pip freeze > "$BUILD_ROOT/linux-dependencies.lock"
{
    printf 'BUILD_UBUNTU=%s\n' "${PRETTY_NAME:-unknown}"
    printf 'BUILD_UBUNTU_VERSION=%s\n' "$VERSION_ID"
    printf 'BUILD_ARCH=%s\n' "$(uname -m)"
    printf 'BUILD_GLIBC=%s\n' "$actual_glibc"
    printf 'BUILD_PYTHON=%s\n' "$("$BUILD_VENV/bin/python" --version 2>&1)"
    printf 'BUILD_PYINSTALLER=%s\n' "$("$BUILD_VENV/bin/python" -m PyInstaller --version)"
} > "$BUILD_ROOT/linux-build-info.txt"

"$TOOLS_DIR/package_release.sh" "$VERSION"
