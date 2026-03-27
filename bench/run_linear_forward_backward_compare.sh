#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SHAPES_CSV="${FA_LINEAR_FB_SHAPES:-bench/linear_shapes.csv}"
WARMUP="${FA_LINEAR_FB_WARMUP:-50}"
ITERS="${FA_LINEAR_FB_ITERS:-200}"
RESULT_DIR="bench/results"
RUN_TS="$(date +"%Y%m%d_%H%M%S")"
NATIVE_CSV="${RESULT_DIR}/native_linear_forward_backward_${RUN_TS}.csv"
TORCH_CSV="${RESULT_DIR}/torch_linear_forward_backward_${RUN_TS}.csv"
COMPARE_CSV="${RESULT_DIR}/linear_forward_backward_compare_${RUN_TS}.csv"

mkdir -p "$RESULT_DIR"

cmake -S . -B build

targets_help="$(cmake --build build --target help 2>/dev/null || true)"
if [[ "$targets_help" != *"linear_forward_backward_bench"* ]]; then
  echo "error: linear_forward_backward_bench target is unavailable." >&2
  echo "hint: install Google Benchmark and reconfigure." >&2
  exit 1
fi

cmake --build build --target linear_forward_backward_bench -j

echo "[1/4] Running native benchmark (Google Benchmark)"
FA_LINEAR_FB_SHAPES="$SHAPES_CSV" FA_LINEAR_FB_WARMUP="$WARMUP" FA_LINEAR_FB_ITERS="$ITERS" \
  ./build/linear_forward_backward_bench \
  --benchmark_out="$NATIVE_CSV" \
  --benchmark_out_format=csv \
  --benchmark_color=false

echo "[2/4] Running PyTorch benchmark"
python3 tools/bench_linear_forward_backward_pytorch.py \
  --shapes "$SHAPES_CSV" \
  --output "$TORCH_CSV" \
  --warmup "$WARMUP" \
  --iters "$ITERS"

echo "[3/4] Comparing results"
python3 tools/compare_linear_forward_backward_bench.py \
  --native "$NATIVE_CSV" \
  --torch "$TORCH_CSV" \
  --output "$COMPARE_CSV"

echo "[4/4] Plotting comparison charts"
if ! python3 tools/plot_linear_forward_backward_compare.py --compare "$COMPARE_CSV" --outdir "$RESULT_DIR"; then
  echo "warning: chart generation failed (install matplotlib to enable plots)." >&2
fi

echo "native:  $NATIVE_CSV"
echo "torch:   $TORCH_CSV"
echo "compare: $COMPARE_CSV"
