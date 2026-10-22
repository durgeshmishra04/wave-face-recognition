#!/usr/bin/env bash
set -euo pipefail

URL="${1:-${WAVE_HEALTH_URL:-http://127.0.0.1:5000/api/cameras}}"
command -v curl >/dev/null 2>&1 || {
    echo "curl is required for the HTTP health check." >&2
    exit 1
}
curl --fail --silent --show-error --max-time 10 "$URL"
printf '\nHTTP endpoint responded: %s\n' "$URL"
