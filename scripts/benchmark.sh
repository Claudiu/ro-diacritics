#!/usr/bin/env bash
# Native inference throughput at the production model shape.
set -euo pipefail
cd "$(dirname "$0")/.."

cargo bench -p diacritics-model --bench restore -- "$@"
