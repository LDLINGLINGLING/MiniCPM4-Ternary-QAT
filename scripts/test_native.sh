#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-${BITNET_BUILD_DIR:-$ROOT/bitnet.cpp/build}}"
BINARY="$BUILD_DIR/bin/test-i2s-scaled"
if [[ ! -x "$BINARY" ]]; then
  printf 'Missing %s; run scripts/build_bitnet.sh first.\n' "$BINARY" >&2
  exit 1
fi
exec "$BINARY"
