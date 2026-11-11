#!/bin/bash
# Unit u63af4bc057 tests: the 212-row port map, the host invariants (half-precision conversions, block dequantizers,
# GGUF header reader, AVX-512 i-quant rows) and the device invariants (block dequant to bf16/f16/f32, S2 plane).
# Exits non-zero if any check fails.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/build/u63af"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${EINSTEINOS_GPUS:-0}}"
echo "== port map =="
python3 tests/u63af/check_port.py "$ROOT"
echo "== host invariants =="
"$OUT/test_u63_host"
echo "== device invariants (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) =="
"$OUT/test_u63_gpu"
echo "unit u63af4bc057: all tests passed"
