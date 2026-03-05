# Tests Runner

Use the local script in this folder to run all GoogleTests or a specific test source.

## Usage

```bash
tests/run.sh
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
```

## Mapping

- `tests/test_tensor.cu` -> `tensor_tests`
- `tests/smoke_main.cpp` -> `flashattn_smoke`
- Fallback: `tests/test_xxx.cu` -> `xxx_tests`

If a new test file uses a different target name, update `tests/run.sh`.

## GoogleTest

`tests/test_tensor.cu` now uses GoogleTest assertions (`ASSERT_*`, `EXPECT_*`) and is discovered by CTest as:

- `TensorCorrectness.AllocFreeStress`
- `TensorCorrectness.H2DAndD2HRoundTrip`
