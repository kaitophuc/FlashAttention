#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage:"
  echo "  tests/run.sh"
  echo "  tests/run.sh <test-source-file> [-- <binary-args...>]"
  echo "  tests/run.sh -- <binary-args...>"
  echo "Examples:"
  echo "  tests/run.sh"
  echo "  tests/run.sh tests/test_tensor.cu"
  echo "  tests/run.sh -- --gtest_filter=TensorCorrectness.H2DAndD2HRoundTrip"
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

target="tensor_tests"
if [[ -n "$source_file" ]]; then
  if [[ ! -f "$source_file" ]]; then
    echo "error: file not found: $source_file" >&2
    usage
    exit 1
  fi

  case "$source_file" in
    tests/*) ;;
    *)
      echo "error: expected file under tests/, got: $source_file" >&2
      usage
      exit 1
      ;;
  esac

  basename_file="$(basename "$source_file")"
  case "$basename_file" in
    test_tensor.cu) target="tensor_tests" ;;
    smoke_main.cpp) target="flashattn_smoke" ;;
    *)
      stem="${basename_file%.*}"
      if [[ "$stem" == test_* ]]; then
        target="${stem#test_}_tests"
      else
        target="$stem"
      fi
      ;;
  esac
fi

cmake -S . -B build
cmake --build build --target "$target" -j
"./build/$target" "${bin_args[@]}"
