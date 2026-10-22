#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"
BUILD="$ROOT/dist/wave-vms/wave-vms"
MODE="${1:-}"

usage() {
    cat <<'USAGE'
Usage: ./run.sh <source|build|compile|status> [mode-specific arguments]

  source   Run app.py using this project's .venv
  build    Run the PyInstaller onedir executable
  compile  Build/rebuild the Linux executable from wave-vms.spec using .venv
  status   Report source environment and compiled build availability
USAGE
}

if [[ -z "$MODE" ]]; then
    usage >&2
    exit 2
fi
shift

source_status() {
    if [[ -x "$PYTHON" ]]; then
        printf 'source: available (%s)\n' "$PYTHON"
    else
        printf 'source: unavailable (missing %s)\n' "$PYTHON"
    fi
}

build_status() {
    if [[ -x "$BUILD" && -d "$ROOT/dist/wave-vms/_internal" ]]; then
        printf 'build: available (%s)\n' "$BUILD"
    elif [[ -x "$BUILD" ]]; then
        printf 'build: executable exists, but PyInstaller _internal directory is missing\n'
    else
        printf 'build: unavailable (run ./run.sh compile on matching Linux)\n'
    fi
}

case "$MODE" in
    source)
        if [[ ! -x "$PYTHON" ]]; then
            echo "Source mode unavailable: $PYTHON does not exist or is not executable." >&2
            exit 1
        fi
        export WAVE_RUNTIME_DIR="${WAVE_RUNTIME_DIR:-$ROOT}"
        cd "$ROOT"
        exec "$PYTHON" "$ROOT/app.py" "$@"
        ;;
    build)
        if [[ ! -x "$BUILD" ]]; then
            echo "Compiled mode unavailable: $BUILD is missing. Run ./run.sh compile on Linux." >&2
            exit 1
        fi
        if [[ ! -d "$ROOT/dist/wave-vms/_internal" ]]; then
            echo "Compiled mode is incomplete: dist/wave-vms/_internal is missing." >&2
            exit 1
        fi
        export WAVE_RUNTIME_DIR="${WAVE_RUNTIME_DIR:-$ROOT}"
        cd "$ROOT"
        exec "$BUILD" "$@"
        ;;
    compile)
        if [[ "$(uname -s)" != Linux ]]; then
            echo "Linux executable builds must run on Linux; Windows PyInstaller output is not Ubuntu-compatible." >&2
            exit 1
        fi
        if [[ ! -x "$PYTHON" ]]; then
            echo "Compile unavailable: create the project environment at $ROOT/.venv first." >&2
            exit 1
        fi
        if [[ ! -f "$ROOT/wave-vms.spec" ]]; then
            echo "Compile unavailable: $ROOT/wave-vms.spec is missing." >&2
            exit 1
        fi
        if ! "$PYTHON" -c 'import PyInstaller' >/dev/null 2>&1; then
            echo "PyInstaller is missing from the project environment. Install it with:" >&2
            echo "  $PYTHON -m pip install -r $ROOT/requirements-build.txt" >&2
            exit 1
        fi
        cd "$ROOT"
        exec "$PYTHON" -m PyInstaller \
            --clean \
            --noconfirm \
            --distpath "$ROOT/dist" \
            --workpath "$ROOT/build/pyinstaller" \
            "$ROOT/wave-vms.spec" "$@"
        ;;
    status)
        source_status
        build_status
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
