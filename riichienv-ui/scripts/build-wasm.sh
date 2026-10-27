#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT_DIR="$SCRIPT_DIR/../src/wasm/pkg"

echo "[build-wasm] Building riichienv-wasm..."
wasm-pack build "$REPO_ROOT/riichienv-wasm" \
  --target web \
  --out-dir "$OUT_DIR"

cp "$REPO_ROOT/riichienv-core/LICENSE.nyanten" "$OUT_DIR/LICENSE.nyanten"

echo "[build-wasm] WASM build complete: $OUT_DIR"
