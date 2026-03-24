# Tests Runner

Use the local script in this folder to run all GoogleTests or a specific test source.
Reusable test helpers are under `tests/include/`.

## Usage

```bash
tests/run.sh
tests/run.sh --label smoke
tests/run.sh --label concurrency
tests/run.sh --label stress
tests/run.sh <test-source-file> [-- <binary-args...>]
tests/run.sh -- <binary-args...>
```

## Examples

```bash
tests/run.sh tests/test_tensor.cu
tests/run.sh tests/smoke_main.cpp
tests/run.sh tests/test_tensor.cu -- --my_arg=value
tests/run.sh -- --gtest_filter=TensorCorrectness.H2DAndD2HRoundTrip
tests/run.sh tests/test_tensor.cu -- --gtest_filter=TensorCorrectness.H2DAndD2HRoundTrip
tests/run.sh --label smoke
tests/run.sh --label concurrency
FA_REQUIRE_CUDA_TESTS=1 tests/run.sh --label smoke
FA_ENABLE_LONG_STRESS=1 tests/run.sh --label stress
```

## Tier Labels

- `smoke`: quick correctness checks and invariants.
- `concurrency`: multi-stream/multi-handle scenarios.
- `stress`: long-running stress tests (including optional long tier).

When selecting by label, `tests/run.sh` maps labels to test-name patterns.

## CI Guardrail

Set `FA_REQUIRE_CUDA_TESTS=1` to fail the run if any selected tests are skipped.
Use this in GPU CI so "tests passed" means tests actually executed.

## Mapping

- `tests/test_tensor.cu` -> `fa_test_test_tensor`
- `tests/smoke_main.cpp` -> `flashattn_smoke`
- Fallback: `tests/test_xxx.cu` -> `fa_test_test_xxx`

If a new test file uses a different target name, update `tests/run.sh`.

## GoogleTest

`tests/test_tensor.cu` now uses GoogleTest assertions (`ASSERT_*`, `EXPECT_*`) and is discovered by CTest as:

- `TensorCorrectness.AllocFreeStress`
- `TensorCorrectness.H2DAndD2HRoundTrip`
