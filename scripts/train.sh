#!/usr/bin/env bash
# Plug and play on a fresh clone: installs the environment, fetches the corpus, trains,
# evaluates and exports to artifacts/export/. Re-running resumes from the last checkpoint.
#
#   git clone <repo> && cd diacritics && scripts/train.sh
#
# Every knob is a DIACRITICS_* env var (see .env.example); flags go to `diacritics train`.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/"; exit 1; }

uv sync --locked
uv run diacritics fetch
uv run diacritics baseline
uv run diacritics train "$@"
uv run diacritics eval

echo "Export ready in artifacts/export/ (config.json + model.safetensors)."
