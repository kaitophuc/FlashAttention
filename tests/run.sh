#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage:"
  echo "  tests/run.sh"
  echo "  tests/run.sh [--label <smoke|concurrency|stress|all>] [--require-cuda-tests] <test-source-file> [-- <binary-args...>]"
  echo "  tests/run.sh [--label <smoke|concurrency|stress|all>] [--require-cuda-tests] -- <binary-args...>"
  echo "Examples:"
  echo "  tests/run.sh"
  echo "  tests/run.sh --label smoke"
  echo "  tests/run.sh --label concurrency"
  echo "  tests/run.sh tests/test_tensor.cu"
  echo "  tests/run.sh tests/test_linear.cu -- --gtest_filter=LinearForward.Invariant*"
  echo "  FA_REQUIRE_CUDA_TESTS=1 tests/run.sh --label smoke"
}

label="${FA_TEST_LABEL:-}"
require_cuda_tests="${FA_REQUIRE_CUDA_TESTS:-0}"
source_file=""
bin_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --label)
      if [[ $# -lt 2 ]]; then
        echo "error: --label requires a value" >&2
        usage
        exit 1
      fi
      label="$2"
      shift 2
      ;;
    --require-cuda-tests)
      require_cuda_tests=1
      shift
      ;;
    --)
      shift
      bin_args=("$@")
      break
      ;;
    *)
      if [[ -z "$source_file" ]]; then
        source_file="$1"
        shift
      else
        echo "error: unexpected argument: $1" >&2
        usage
        exit 1
      fi
      ;;
  esac
done

if [[ -n "$label" ]]; then
  case "$label" in
    smoke|concurrency|stress|all) ;;
    *)
      echo "error: unsupported label '$label'. Expected smoke|concurrency|stress|all." >&2
      exit 1
      ;;
  esac
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

contains_gtest_filter=0
for arg in "${bin_args[@]}"; do
  if [[ "$arg" == --gtest_filter=* ]]; then
    contains_gtest_filter=1
    break
  fi
done

if [[ -n "$label" && "$label" != "all" && -n "$target" ]]; then
  if [[ $contains_gtest_filter -eq 1 ]]; then
    echo "error: --label cannot be combined with explicit --gtest_filter in binary args" >&2
    exit 1
  fi

  case "$label" in
    smoke)
      bin_args+=("--gtest_filter=LinearForward.Rejects*:LinearForward.SweepAllCases:LinearForward.Numeric*:LinearForward.Invariant*:TensorCorrectness.*")
      ;;
    concurrency)
      bin_args+=("--gtest_filter=LinearForward.MultiStreamConcurrencyIndependentWorkloads:LinearForward.ConcurrencyMatrix_*")
      ;;
    stress)
      bin_args+=("--gtest_filter=LinearForward.Stress_*:LinearForward.SingleStreamOrderingReuseStress*")
      ;;
  esac
fi

check_required_cuda_execution() {
  local log_file="$1"
  local mode="$2"

  local skipped_count=0
  local run_count=0

  if [[ "$mode" == "ctest" ]]; then
    skipped_count=$(grep -E -c '\*\*\*Skipped' "$log_file" || true)
    run_count=$(grep -E -c 'Start [0-9]+:' "$log_file" || true)
  else
    skipped_count=$(grep -E -c '\[  SKIPPED \]' "$log_file" || true)
    run_count=$(grep -E -c '\[ RUN      \]' "$log_file" || true)
  fi

  if (( run_count == 0 )); then
    echo "error: no tests executed." >&2
    exit 1
  fi

  if [[ "$require_cuda_tests" != "1" ]]; then
    return
  fi

  if (( skipped_count > 0 )); then
    echo "error: FA_REQUIRE_CUDA_TESTS=1 but $skipped_count test(s) were skipped." >&2
    exit 1
  fi
}

# Allow overriding CUDA arch list, but default to native to avoid stale or mismatched
# cache values across machines/driver updates.
: "${FA_CUDA_ARCHITECTURES:=native}"
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="${FA_CUDA_ARCHITECTURES}"
if [[ -z "$target" ]]; then
  if [[ ${#bin_args[@]} -gt 0 ]]; then
    echo "error: binary args require an explicit test source file" >&2
    usage
    exit 1
  fi

  # Build all targets first so gtest_discover_tests entries have backing executables.
  cmake --build build -j

  ctest_cmd=(ctest --test-dir build --output-on-failure)
  if [[ -n "$label" && "$label" != "all" ]]; then
    case "$label" in
      smoke)
        ctest_cmd+=( -R "(Rejects|SweepAllCases|Numeric|Invariant|SingleStream|TensorCorrectness\\.)" )
        ;;
      concurrency)
        ctest_cmd+=( -R "(Concurrency|ConcurrencyMatrix|CublasHandleMatrix)" )
        ;;
      stress)
        ctest_cmd+=( -R "(Stress_|ReuseStress)" )
        ;;
    esac
  fi

  tmp_log="$(mktemp)"
  trap 'rm -f "$tmp_log"' EXIT
  set +e
  "${ctest_cmd[@]}" | tee "$tmp_log"
  ctest_status=${PIPESTATUS[0]}
  set -e

  if [[ $ctest_status -ne 0 ]]; then
    exit $ctest_status
  fi

  check_required_cuda_execution "$tmp_log" "ctest"
else
  cmake --build build --target "$target" -j

  tmp_log="$(mktemp)"
  trap 'rm -f "$tmp_log"' EXIT
  set +e
  "./build/$target" "${bin_args[@]}" | tee "$tmp_log"
  test_status=${PIPESTATUS[0]}
  set -e

  if [[ $test_status -ne 0 ]]; then
    exit $test_status
  fi

  check_required_cuda_execution "$tmp_log" "gtest"
fi
