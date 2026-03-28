# Benchmarks

This folder contains:

- native CUDA benchmark binaries (Google Benchmark),
- runner scripts for building/executing benchmarks,
- PyTorch comparison and plotting scripts.

## Quick Run

```bash
bench/run.sh
```

By default this runs `tensor_bandwidth_bench`.

Run a specific benchmark source:

```bash
bench/run.sh bench/linear_forward_bench.cu
bench/run.sh bench/linear_backward_bench.cu
bench/run.sh bench/linear_forward_backward_bench.cu
```

Pass benchmark args through:

```bash
bench/run.sh -- --benchmark_filter=BM_CopyD2D
bench/run.sh bench/tensor_bandwidth_bench.cu -- --benchmark_filter=BM_CopyD2D
```

## Runner Behavior

`bench/run.sh` does the following:

- configures CMake with:
  - `FA_BUILD_BENCHMARKS=ON`
  - `FA_BUILD_TESTS=OFF`
  - `FA_BUILD_PYTHON=OFF`
  - `FA_AUTO_FETCH_BENCHMARK` (default `ON`)
- checks target availability via `cmake --build build --target help`
- builds selected target
- executes `./build/<target>`

Default and common environment knobs:

- `CMAKE_BUILD_TYPE` (default `Release`)
- `FA_AUTO_FETCH_BENCHMARK` (default `ON`)

## Compare Workflows (Native vs PyTorch)

Forward:

```bash
bench/run_linear_compare.sh
```

Backward:

```bash
bench/run_linear_backward_compare.sh
```

Forward + Backward:

```bash
bench/run_linear_forward_backward_compare.sh
```

All compare scripts:

- create timestamped CSVs under `bench/results/`,
- run native benchmark binary,
- run PyTorch baseline script,
- generate merged comparison CSV,
- try to render plots (warn if `matplotlib` is missing).

## Compare Script Knobs

Forward script (`run_linear_compare.sh`):

- `FA_LINEAR_SHAPES` (default `bench/linear_shapes.csv`)
- `FA_LINEAR_WARMUP` (default `50`)
- `FA_LINEAR_ITERS` (default `200`)

Backward script (`run_linear_backward_compare.sh`):

- `FA_LINEAR_BWD_SHAPES` (default `bench/linear_shapes.csv`)
- `FA_LINEAR_BWD_WARMUP` (default `50`)
- `FA_LINEAR_BWD_ITERS` (default `200`)

Forward+Backward script (`run_linear_forward_backward_compare.sh`):

- `FA_LINEAR_FB_SHAPES` (default `bench/linear_shapes.csv`)
- `FA_LINEAR_FB_WARMUP` (default `50`)
- `FA_LINEAR_FB_ITERS` (default `200`)
