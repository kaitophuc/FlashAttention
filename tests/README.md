# Tests

This folder contains:

- C++ GoogleTest sources (`tests/test_*.cu`),
- shared C++ test helpers (`tests/include/`),
- Python API tests (`tests/python/test_*.py`),
- runner scripts (`tests/run.sh`, `tests/run_python_tests.sh`).

## C++ Tests (`tests/run.sh`)

Run all discovered tests:

```bash
tests/run.sh
```

Run selected groups:

```bash
tests/run.sh --label smoke
tests/run.sh --label stress
```

Run one source file target:

```bash
tests/run.sh tests/test_tensor.cu
tests/run.sh tests/test_linear.cu -- --gtest_filter=LinearForward.Invariant*
```

Behavior notes:

- Script configures with `cmake -S . -B build`.
- Script sets default `FA_CUDA_ARCHITECTURES=120` unless overridden.
- If no source file is provided, it runs `ctest` on discovered tests.
- If a source file is provided, it builds and runs the mapped test binary directly.

## Python API Tests (`tests/run_python_tests.sh`)

Run:

```bash
tests/run_python_tests.sh
```

What it does:

- resolves repo root from script location,
- installs editable package (`pip install -e <repo-root>`),
- sets `PYTHONPATH=<repo-root>/python`,
- runs unittest discovery in `tests/python`.

Optional interpreter override:

```bash
PYTHON_BIN=./venv/bin/python tests/run_python_tests.sh
```

## CMake Target

`CMakeLists.txt` defines a convenience target:

```bash
cmake --build build --target fa_python_tests
```

This runs `tests/run_python_tests.sh`.

## Guardrails and Labels

- Set `FA_REQUIRE_CUDA_TESTS=1` to fail if selected tests are skipped.
- Labels:
  - `smoke`: fast correctness/invariant-oriented patterns (including `MultiStream.*`).
  - `stress`: long reuse/order stress patterns (including `MultiStreamStress.*`).
