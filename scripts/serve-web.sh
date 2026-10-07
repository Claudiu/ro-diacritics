#!/usr/bin/env bash
# Serves the demo page locally; WASM modules need http://, not file://.
set -euo pipefail
cd "$(dirname "$0")/../web"

port="${1:-8000}"
echo "http://localhost:$port/"
exec python3 -m http.server "$port" --bind 127.0.0.1
