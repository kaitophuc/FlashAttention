# FlashAttention Scratch / `ktorch`

This repository is a CUDA-first implementation playground with:

- a C++ tensor runtime and CUDA ops,
- C++ correctness tests and benchmarks,
- a Python package (`ktorch`) backed by pybind11.

The current codebase is runnable for tensor/ops correctness, benchmark workflows, and tiny Python MLP smoke training.

## Current Layout

- `include/`: public C++ headers (`tensor.h`, `ops.h`, stream/allocator helpers).
- `src/`: CUDA/C++ implementations (`ops_linear.cu`, `ops_layernorm.cu`, `ops_activation.cu`, `ops_softmax.cu`).
- `bindings/python/`: pybind11 bindings for `ktorch._C`.
- `python/ktorch/`: Python API surface (`Tensor`, `Device`, `DType`, `ops`).
- `tests/`: C++ tests, Python API tests, and test runner scripts.
- `bench/`: native benchmark sources and PyTorch comparison tooling.
- `tools/ktorch/`: runnable Python examples (`mlp_manual_train.py`).

## Current Feature Matrix

Implemented and wired:

- Tensor operations: metadata, `numel`, `nbytes`, `view`, `clone`, `zero_`, host list copy helpers.
- Ops in both C++ and Python: `linear`, `layernorm`, `relu`, `softmax`.
- Optimizer update op in both C++ and Python: in-place `sgd_update_` (CUDA F32).
- C++ GoogleTests for tensor/linear/layernorm/relu/softmax.
- Python unittests under `tests/python/`.
- Google Benchmark targets and compare scripts under `bench/`.

Present but not fully wired:

- Dropout API is declared in C++ headers, but `src/ops_dropout.cu` and `tests/test_dropout.cu` are placeholders.
- Dropout is not bound in Python yet.

Practical limitations right now:

- Python host I/O helpers are list-based (`copy_from_list_float`, `to_list_float`) and F32-oriented.
- `sgd_update_` currently supports CUDA `F32` tensors only.
- Most Python smoke/training flows are CUDA-focused.
- There is no autograd engine yet; backward is called manually.

## Requirements

- Linux with working NVIDIA driver + CUDA toolkit.
- CMake 3.24+.
- C++17 + CUDA toolchain.
- Python 3.10+ for package workflows.

## Detailed Quick Start

### 1. Build C++ targets

```bash
cmake -S . -B build
cmake --build build -j
```

### 2. Run C++ tests

```bash
tests/run.sh
```

Useful variants:

```bash
tests/run.sh --label smoke
tests/run.sh --label stress
tests/run.sh tests/test_linear.cu
tests/run.sh tests/test_tensor.cu -- --gtest_filter=TensorCorrectness.H2DAndD2HRoundTrip
```

Notes:

- `tests/run.sh` configures CMake automatically.
- It defaults `FA_CUDA_ARCHITECTURES=120` unless you override it.
- Set `FA_REQUIRE_CUDA_TESTS=1` to fail if selected tests are skipped.

### 3. Run Python API tests

```bash
tests/run_python_tests.sh
```

What this script does:

- installs editable package (`pip install -e <repo-root>`),
- sets `PYTHONPATH=<repo-root>/python`,
- runs `python -m unittest discover -s tests/python -p 'test_*.py' -v`.

You can override interpreter:

```bash
PYTHON_BIN=./venv/bin/python tests/run_python_tests.sh
```

You can also run through CMake target:

```bash
cmake --build build --target fa_python_tests
```

### 4. Run tiny MLP manual training smoke

```bash
python3 -m pip install -e .
python3 tools/ktorch/mlp_manual_train.py
```

Expected output shape:

- prints step-0 and final-step loss,
- final loss should decrease,
- prints final `inference_output` list.

### 5. Run FashionMNIST manual-vs-PyTorch helpers

Use the dedicated runner:

```bash
tools/ktorch/run_fashion_mnist_compare.sh --mode both --benchmark-compare --batch-size 128
```

Run only the ktorch path:

```bash
tools/ktorch/run_fashion_mnist_compare.sh --mode manual --epochs 3 --max-train-batches 200
```

## Build Profiles

### C++ test/bench profile

```bash
cmake -S . -B build \
  -DFA_BUILD_TESTS=ON \
  -DFA_BUILD_BENCHMARKS=ON \
  -DFA_BUILD_PYTHON=OFF
cmake --build build -j
```

### Python extension profile (manual CMake)

```bash
cmake -S . -B build_py \
  -DFA_BUILD_PYTHON=ON \
  -DFA_BUILD_TESTS=OFF \
  -DFA_BUILD_BENCHMARKS=OFF
cmake --build build_py -j
```

### Python package profile (recommended)

`pyproject.toml` uses scikit-build-core and sets:

- `FA_BUILD_PYTHON=ON`
- `FA_BUILD_TESTS=OFF`
- `FA_BUILD_BENCHMARKS=OFF`

Use:

```bash
python3 -m pip install -e .
```

## Benchmarks

Run the native benchmark runner:

```bash
bench/run.sh
```

Run native-vs-PyTorch compare workflows:

```bash
bench/run_linear_compare.sh
bench/run_linear_backward_compare.sh
bench/run_linear_forward_backward_compare.sh
```

See `bench/README.md` for all options and env knobs.

## Additional Docs

- Test guide: `tests/README.md`
- Benchmark guide: `bench/README.md`
- Implementation notes: `docs/ktorch_implementation_notes.md`
