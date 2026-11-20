#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="$ROOT/bitnet.cpp"
BUILD_DIR="${BITNET_BUILD_DIR:-$SOURCE_DIR/build}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '8')}"

cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DLLAMA_BUILD_COMMON=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  "$@"
cmake --build "$BUILD_DIR" \
  --target llama-cli llama-quantize llama-bitnet-eval llama-bench test-i2s-scaled \
  -j "$JOBS"
