#!/usr/bin/env bash
# Everything CI runs: lint, types, tests, both languages.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -q

cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
