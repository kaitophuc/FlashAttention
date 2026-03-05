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
bench/run.sh bench/tensor_bandwidth_bench.cu -- --iters=50
bench/run.sh bench/tensor_bandwidth_bench.cu -- --benchmark_filter=BM_CopyD2D
bench/run.sh -- --benchmark_filter=BM_CopyD2D
```

## Mapping

- `bench/tensor_bandwidth_bench.cu` -> `tensor_bandwidth_bench`
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
