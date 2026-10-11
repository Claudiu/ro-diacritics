#!/usr/bin/env bash
# Assembles the browser extension for Chrome and Firefox into build/extension/<browser>/
# (loadable unpacked) and build/extension/diacritice-<browser>.zip (for the stores).
# Reuses the WASM bundle and model from scripts/build-wasm.sh, building them if missing.
# VERSION=1.2.3 overrides the manifest version (CI sets it from the release tag).
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f web/pkg/diacritics_wasm_bg.wasm ] && [ -f web/model/model.safetensors ] || scripts/build-wasm.sh

for browser in chrome firefox; do
  out="build/extension/$browser"
  rm -rf "$out"
  mkdir -p "$out/pkg" "$out/model"
  cp extension/background.js extension/content.js "$out/"
  cp -r extension/icons "$out/"
  cp web/pkg/diacritics_wasm.js web/pkg/diacritics_wasm_bg.wasm "$out/pkg/"
  cp web/model/config.json web/model/model.safetensors "$out/model/"
  # One manifest; Firefox has no background service workers yet and wants an add-on id.
  python3 - "$browser" extension/manifest.json "$out/manifest.json" <<'EOF'
import json, os, sys
browser, src, dst = sys.argv[1:]
m = json.load(open(src))
m["version"] = os.environ.get("VERSION") or m["version"]
if browser == "firefox":
    m["background"] = {"scripts": ["background.js"], "type": "module"}
    m["browser_specific_settings"] = {"gecko": {
        "id": "diacritice@ro-diacritics",
        "strict_min_version": "140.0",
        "data_collection_permissions": {"required": ["none"]},
    }}
json.dump(m, open(dst, "w"), ensure_ascii=False, indent=2)
EOF
  rm -f "build/extension/diacritice-$browser.zip"
  (cd "$out" && zip -qr "../diacritice-$browser.zip" .)
done

ls -la build/extension/*.zip
