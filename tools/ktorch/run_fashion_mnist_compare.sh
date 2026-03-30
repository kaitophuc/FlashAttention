#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage:"
  echo "  tools/ktorch/run_fashion_mnist_compare.sh [--mode manual|pytorch|both] [script args ...]"
  echo ""
  echo "Examples:"
  echo "  tools/ktorch/run_fashion_mnist_compare.sh --mode both --benchmark-compare --batch-size 128"
  echo "  tools/ktorch/run_fashion_mnist_compare.sh --mode manual --epochs 3 --max-train-batches 200"
  echo ""
  echo "All trailing args are passed through to training scripts."
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODE="both"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --mode)
      if [[ $# -lt 2 ]]; then
        echo "error: --mode requires a value" >&2
        exit 1
      fi
      MODE="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

case "$MODE" in
  manual|pytorch|both) ;;
  *)
    echo "error: unsupported mode '$MODE' (expected manual|pytorch|both)" >&2
    exit 1
    ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
  if [[ -x "${REPO_ROOT}/venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/venv/bin/python"
  else
    echo "error: pip unavailable for ${PYTHON_BIN} and ${REPO_ROOT}/venv/bin/python not found" >&2
    exit 1
  fi
fi

"$PYTHON_BIN" -m pip install -e "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}"

run_manual() {
  echo "[run] fashion_mnist_manual_train.py ${EXTRA_ARGS[*]}"
  "$PYTHON_BIN" "${REPO_ROOT}/tools/ktorch/fashion_mnist_manual_train.py" "${EXTRA_ARGS[@]}"
}

run_pytorch() {
  echo "[run] fashion_mnist_pytorch_train.py ${EXTRA_ARGS[*]}"
  "$PYTHON_BIN" "${REPO_ROOT}/tools/ktorch/fashion_mnist_pytorch_train.py" "${EXTRA_ARGS[@]}"
}

case "$MODE" in
  manual)
    run_manual
    ;;
  pytorch)
    run_pytorch
    ;;
  both)
    run_manual
    run_pytorch
    ;;
esac
