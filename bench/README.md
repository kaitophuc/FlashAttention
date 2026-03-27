# Bench Runner

Use the local script in this folder to build and run a specific benchmark source.

Benchmark runtime is powered by Google Benchmark.
If missing, CMake auto-fetches it by default (`FA_AUTO_FETCH_BENCHMARK=ON`).

## Usage

```bash
bench/run.sh
bench/run.sh <bench-source-file> [-- <binary-args...>]
bench/run.sh -- <binary-args...>
```

## Examples

```bash
bench/run.sh bench/tensor_bandwidth_bench.cu
bench/run.sh bench/linear_forward_bench.cu
bench/run.sh bench/linear_backward_bench.cu
bench/run.sh bench/linear_forward_backward_bench.cu
bench/run.sh bench/tensor_bandwidth_bench.cu -- --iters=50
bench/run.sh bench/tensor_bandwidth_bench.cu -- --benchmark_filter=BM_CopyD2D
bench/run.sh -- --benchmark_filter=BM_CopyD2D
```

## Mapping

- `bench/tensor_bandwidth_bench.cu` -> `tensor_bandwidth_bench`
- `bench/linear_forward_bench.cu` -> `linear_forward_bench`
- `bench/linear_backward_bench.cu` -> `linear_backward_bench`
- `bench/linear_forward_backward_bench.cu` -> `linear_forward_backward_bench`
- Fallback: `bench/xxx.cu` -> `xxx`

If a new benchmark file uses a different target name, update `bench/run.sh`.

## Dependency Behavior

Default behavior:

1. CMake tries `find_package(benchmark)`.
2. If not found, CMake fetches Google Benchmark source automatically.

To disable auto-fetch:

```bash
cmake -S . -B build -DFA_AUTO_FETCH_BENCHMARK=OFF
```

If auto-fetch is disabled, install system package manually (e.g. `libbenchmark-dev`).

## Linear vs PyTorch Compare

Run native CUDA linear forward benchmark and compare with PyTorch:

```bash
bench/run_linear_compare.sh
```

Outputs:

- `bench/results/native_linear.csv`
- `bench/results/torch_linear.csv`
- `bench/results/linear_compare.csv`

Config knobs (optional environment variables):

- `FA_LINEAR_SHAPES` (default `bench/linear_shapes.csv`)
- `FA_LINEAR_WARMUP` (default `50`)
- `FA_LINEAR_ITERS` (default `200`)

## Linear Backward vs PyTorch Compare

Run native CUDA linear backward benchmark and compare with PyTorch:

```bash
bench/run_linear_backward_compare.sh
```

Outputs:

- `bench/results/native_linear_backward_<timestamp>.csv`
- `bench/results/torch_linear_backward_<timestamp>.csv`
- `bench/results/linear_backward_compare_<timestamp>.csv`

Config knobs (optional environment variables):

- `FA_LINEAR_BWD_SHAPES` (default `bench/linear_shapes.csv`)
- `FA_LINEAR_BWD_WARMUP` (default `50`)
- `FA_LINEAR_BWD_ITERS` (default `200`)

## Linear Forward+Backward vs PyTorch Compare

Run native CUDA linear forward+backward benchmark and compare with PyTorch:

```bash
bench/run_linear_forward_backward_compare.sh
```

Outputs:

- `bench/results/native_linear_forward_backward_<timestamp>.csv`
- `bench/results/torch_linear_forward_backward_<timestamp>.csv`
- `bench/results/linear_forward_backward_compare_<timestamp>.csv`

Config knobs (optional environment variables):

- `FA_LINEAR_FB_SHAPES` (default `bench/linear_shapes.csv`)
- `FA_LINEAR_FB_WARMUP` (default `50`)
- `FA_LINEAR_FB_ITERS` (default `200`)
