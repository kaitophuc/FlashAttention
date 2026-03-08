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

target=""
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
    smoke_main.cpp) target="flashattn_smoke" ;;
    *)
      stem="${source_file#tests/}"
      stem="${stem%.*}"
      stem="${stem//\//_}"
      if [[ "$stem" == test_* ]]; then
        target="fa_test_${stem}"
      else
        target="$stem"
      fi
      ;;
  esac
fi

cmake -S . -B build
if [[ -z "$target" ]]; then
  if [[ ${#bin_args[@]} -gt 0 ]]; then
    echo "error: binary args require an explicit test source file" >&2
    usage
    exit 1
  fi
  # Build all targets first so gtest_discover_tests entries have backing executables.
  cmake --build build -j
  ctest --test-dir build --output-on-failure
else
  cmake --build build --target "$target" -j
  "./build/$target" "${bin_args[@]}"
fi
