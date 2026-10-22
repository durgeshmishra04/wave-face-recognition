#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET_UBUNTU_RELEASE="${TARGET_UBUNTU_RELEASE:-}"
TARGET_ARCH="${TARGET_ARCH:-}"
TARGET_GLIBC_VERSION="${TARGET_GLIBC_VERSION:-}"
BUILD_VENV="${BUILD_VENV:-$ROOT/.build-venv}"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Build must run on Linux; Windows-built executables do not run on Ubuntu." >&2
    exit 1
fi
if [[ -z "$TARGET_UBUNTU_RELEASE" ]]; then
    echo "Set TARGET_UBUNTU_RELEASE to the target VM's Ubuntu VERSION_ID (for example 22.04)." >&2
    exit 1
fi
if [[ -z "$TARGET_ARCH" || -z "$TARGET_GLIBC_VERSION" ]]; then
    echo "Set TARGET_ARCH and TARGET_GLIBC_VERSION from the target VM before building." >&2
    exit 1
fi
if [[ ! -r /etc/os-release ]]; then
    echo "Cannot determine the Linux distribution from /etc/os-release." >&2
    exit 1
fi

source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "$TARGET_UBUNTU_RELEASE" ]]; then
    echo "Build host must be Ubuntu $TARGET_UBUNTU_RELEASE; found ${PRETTY_NAME:-unknown}." >&2
    exit 1
fi
if [[ "$(uname -m)" != "$TARGET_ARCH" ]]; then
    echo "Build architecture $(uname -m) does not match target $TARGET_ARCH." >&2
    exit 1
fi
if [[ "$(getconf GNU_LIBC_VERSION 2>/dev/null || echo unknown)" != "$TARGET_GLIBC_VERSION" ]]; then
    echo "Build glibc does not match target $TARGET_GLIBC_VERSION." >&2
    exit 1
fi
if ! command -v python3.11 >/dev/null 2>&1; then
    echo "Python 3.11 is required (the inspected project environment uses Python 3.11)." >&2
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

python3.11 -m venv "$BUILD_VENV"
"$BUILD_VENV/bin/python" -m pip install --upgrade pip
"$BUILD_VENV/bin/python" -m pip install \
    -r "$ROOT/requirements.txt" \
    -r "$ROOT/requirements-build.txt"
"$BUILD_VENV/bin/python" -m PyInstaller \
    --clean \
    --noconfirm \
    --distpath "$ROOT/dist" \
    --workpath "$ROOT/build/pyinstaller" \
    "$ROOT/wave-vms.spec"

mkdir -p "$ROOT/build"
"$BUILD_VENV/bin/python" -m pip freeze > "$ROOT/build/linux-dependencies.lock"
{
    printf 'BUILD_UBUNTU=%s\n' "${PRETTY_NAME:-unknown}"
    printf 'BUILD_UBUNTU_VERSION=%s\n' "$VERSION_ID"
    printf 'BUILD_ARCH=%s\n' "$(uname -m)"
    printf 'BUILD_GLIBC=%s\n' "$(getconf GNU_LIBC_VERSION 2>/dev/null || echo unknown)"
    printf 'BUILD_PYTHON=%s\n' "$("$BUILD_VENV/bin/python" --version 2>&1)"
    printf 'BUILD_PYINSTALLER=%s\n' "$("$BUILD_VENV/bin/python" -m PyInstaller --version)"
} | tee "$ROOT/build/linux-build-info.txt"
echo "PyInstaller output: $ROOT/dist/wave-vms"
