#!/usr/bin/env bash
# Builds the browser bundle into web/pkg/ and copies the exported model to web/model/.
set -euo pipefail
cd "$(dirname "$0")/.."

export_dir="${1:-artifacts/export}"

command -v wasm-pack >/dev/null || { echo "wasm-pack is required: cargo install wasm-pack"; exit 1; }

wasm-pack build crates/wasm --release --target web --out-dir ../../web/pkg --no-typescript

if [ -f "$export_dir/config.json" ]; then
  mkdir -p web/model
  cp "$export_dir/config.json" "$export_dir/model.safetensors" web/model/
else
  echo "No export in $export_dir; the page will fail to load a model until one exists."
fi

echo "Built. Serve with: scripts/serve-web.sh"
ls -la web/pkg/*.wasm
