#!/usr/bin/env bash
# Regenerates the tiny PyTorch parity fixture and checks the candle forward pass against it.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run diacritics fixture test/fixtures/parity
cargo test -p diacritics-model --release
