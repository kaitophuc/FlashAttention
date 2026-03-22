#include "cu_stream.h"
#include "tensor.h"

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>

namespace {

bool cuda_available() {
    int device_count = 0;
    const cudaError_t cuda_ok = cudaGetDeviceCount(&device_count);
    return cuda_ok == cudaSuccess && device_count > 0;
}

double measure_copy_from_seconds(Tensor& dst, const Tensor& src, Stream& stream) {
    Event start;
    Event stop;
    record(start, stream);
    dst.copy_from(src, stream);
    record(stop, stream);
    stop.synchronize();
    return static_cast<double>(elapsed_time(start, stop)) * 1.0e-3;
}

void run_copy_benchmark(benchmark::State& state, cudaMemcpyKind kind) {
    if (kind != cudaMemcpyHostToHost && !cuda_available()) {
        state.SkipWithError("CUDA device unavailable");
        return;
    }

    const size_t bytes = static_cast<size_t>(state.range(0));
    const int64_t n = static_cast<int64_t>(bytes);
    Stream stream;

    Tensor h_src = Tensor::zeros({n}, DType::U8, Device::CPU);
    Tensor h_dst = Tensor::zeros({n}, DType::U8, Device::CPU);
    Tensor d_src = Tensor::zeros({n}, DType::U8, Device::CUDA);
    Tensor d_dst = Tensor::zeros({n}, DType::U8, Device::CUDA);

    Tensor* src = nullptr;
    Tensor* dst = nullptr;
    if (kind == cudaMemcpyHostToDevice) {
        src = &h_src;
        dst = &d_dst;
    } else if (kind == cudaMemcpyDeviceToHost) {
        src = &d_src;
        dst = &h_dst;
    } else if (kind == cudaMemcpyHostToHost) {
        src = &h_src;
        dst = &h_dst;
    } else {
        src = &d_src;
        dst = &d_dst;
    }

    // Warm-up before timed iterations.
    dst->copy_from(*src, stream);
    stream.synchronize();

    double total_seconds = 0.0;
    for (auto _ : state) {
        (void)_;
        const double seconds = measure_copy_from_seconds(*dst, *src, stream);
        total_seconds += seconds;
        state.SetIterationTime(seconds);
    }

    const double iters = static_cast<double>(state.iterations());
    if (iters > 0.0 && total_seconds > 0.0) {
        const double avg_seconds = total_seconds / iters;
        const double gbps = (static_cast<double>(bytes) / 1.0e9) / avg_seconds;
        state.counters["GB/s"] = gbps;
    }
    state.counters["Bytes"] = static_cast<double>(bytes);
}

void BM_CopyH2D(benchmark::State& state) {
    run_copy_benchmark(state, cudaMemcpyHostToDevice);
}

void BM_CopyD2H(benchmark::State& state) {
    run_copy_benchmark(state, cudaMemcpyDeviceToHost);
}

void BM_CopyD2D(benchmark::State& state) {
    run_copy_benchmark(state, cudaMemcpyDeviceToDevice);
}

void BM_CopyH2H(benchmark::State& state) {
    run_copy_benchmark(state, cudaMemcpyHostToHost);
}

}  // namespace

BENCHMARK(BM_CopyH2D)->Arg(256ULL << 20)->UseManualTime();
BENCHMARK(BM_CopyD2H)->Arg(256ULL << 20)->UseManualTime();
BENCHMARK(BM_CopyD2D)->Arg(256ULL << 20)->UseManualTime();
BENCHMARK(BM_CopyH2H)->Arg(256ULL << 20)->UseManualTime();

BENCHMARK_MAIN();
