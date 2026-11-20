#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  printf 'Usage: %s CHECKPOINT RUN_CONFIG {row|g32} OUTPUT_DIR [BASE_MODEL]\n' "$0" >&2
  exit 2
fi

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="$1"
RUN_CONFIG="$2"
GRANULARITY="$3"
OUTPUT_DIR="$4"
BASE_MODEL="${5:-$ROOT/MiniCPM4-0.5B-fjzcuwlf}"
LLAMA_DIR="$ROOT/bitnet.cpp/3rdparty/llama.cpp"
BUILD_DIR="${BITNET_BUILD_DIR:-$ROOT/bitnet.cpp/build}"
PYTHON_BIN="${PYTHON:-python}"

case "$GRANULARITY" in
  row) GGML_TYPE=I2_S_ROW ;;
  g32) GGML_TYPE=I2_S_G32 ;;
  *) printf 'Granularity must be row or g32, got %s\n' "$GRANULARITY" >&2; exit 2 ;;
esac
if [[ -e "$OUTPUT_DIR" ]]; then
  printf 'Refusing to overwrite existing output: %s\n' "$OUTPUT_DIR" >&2
  exit 1
fi
if [[ ! -x "$BUILD_DIR/bin/llama-quantize" ]]; then
  printf 'Missing llama-quantize; run scripts/build_bitnet.sh first.\n' >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
HF_DIR="$OUTPUT_DIR/hf_fp32_master"
F32_GGUF="$OUTPUT_DIR/model-F32.gguf"
NATIVE_GGUF="$OUTPUT_DIR/model-$GGML_TYPE.gguf"

"$PYTHON_BIN" "$ROOT/tools/export_bitnet_hf.py" \
  --checkpoint "$CHECKPOINT" \
  --run-config "$RUN_CONFIG" \
  --base-model "$BASE_MODEL" \
  --output-dir "$HF_DIR" \
  --granularity "$GRANULARITY"

PYTHONPATH="$LLAMA_DIR/gguf-py${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" "$LLAMA_DIR/convert_hf_to_gguf.py" "$HF_DIR" \
  --outfile "$F32_GGUF" --outtype f32

"$BUILD_DIR/bin/llama-quantize" \
  "$F32_GGUF" "$NATIVE_GGUF" "$GGML_TYPE" "${QUANTIZE_THREADS:-16}"

PYTHONPATH="$LLAMA_DIR/gguf-py${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" "$ROOT/tools/verify_bitnet_gguf.py" \
  --gguf "$NATIVE_GGUF" \
  --hf-dir "$HF_DIR" \
  --granularity "$GRANULARITY" \
  --output "$OUTPUT_DIR/roundtrip.json"

printf 'Native GGUF: %s\n' "$NATIVE_GGUF"
