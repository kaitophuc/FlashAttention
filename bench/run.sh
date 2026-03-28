#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage:"
  echo "  bench/run.sh"
  echo "  bench/run.sh <bench-source-file> [-- <binary-args...>]"
  echo "  bench/run.sh -- <binary-args...>"
  echo "Examples:"
  echo "  bench/run.sh"
  echo "  bench/run.sh bench/tensor_bandwidth_bench.cu"
  echo "  bench/run.sh bench/linear_forward_bench.cu"
  echo "  bench/run.sh bench/linear_backward_bench.cu"
  echo "  bench/run.sh bench/linear_forward_backward_bench.cu"
  echo "  bench/run.sh -- --benchmark_filter=BM_CopyD2D"
}

source_file=""
if [[ $# -gt 0 && "$1" != "--" ]]; then
  source_file="$1"
  shift
fi

bin_args=()
if [[ $# -gt 0 ]]; then
  if [[ "$1" == "--" ]]; then
    shift
  fi
  bin_args=("$@")
fi

target="tensor_bandwidth_bench"
if [[ -n "$source_file" ]]; then
  if [[ ! -f "$source_file" ]]; then
    echo "error: file not found: $source_file" >&2
    usage
    exit 1
  fi

  case "$source_file" in
    bench/*) ;;
    *)
      echo "error: expected file under bench/, got: $source_file" >&2
      usage
      exit 1
      ;;
  esac

  basename_file="$(basename "$source_file")"
  case "$basename_file" in
    tensor_bandwidth_bench.cu) target="tensor_bandwidth_bench" ;;
    *)
      target="${basename_file%.*}"
      ;;
  esac
fi

CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
FA_AUTO_FETCH_BENCHMARK="${FA_AUTO_FETCH_BENCHMARK:-ON}"
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
  -DFA_BUILD_BENCHMARKS=ON \
  -DFA_BUILD_TESTS=OFF \
  -DFA_BUILD_PYTHON=OFF \
  -DFA_AUTO_FETCH_BENCHMARK="${FA_AUTO_FETCH_BENCHMARK}"
targets_help="$(cmake --build build --target help 2>/dev/null || true)"
if [[ "$targets_help" != *"$target"* ]]; then
  echo "error: benchmark target '$target' is unavailable." >&2
  echo "hint: ensure Google Benchmark is installed or set FA_AUTO_FETCH_BENCHMARK=ON." >&2
  exit 1
fi
cmake --build build --target "$target" -j
"./build/$target" "${bin_args[@]}"
