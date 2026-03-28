#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
  if [[ -x "${REPO_ROOT}/venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/venv/bin/python"
  else
    echo "error: pip is unavailable for ${PYTHON_BIN} and ${REPO_ROOT}/venv/bin/python not found" >&2
    exit 1
  fi
fi

"$PYTHON_BIN" -m pip install -e "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}"
"$PYTHON_BIN" -m unittest discover -s "${REPO_ROOT}/tests/python" -p 'test_*.py' -v
