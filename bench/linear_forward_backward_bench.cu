#include "ops.h"

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct LinearCase {
    std::string name;
    int m;
    int k;
    int n;
    bool bias;
};

int env_int(const char* key, int default_value) {
    const char* raw = std::getenv(key);
    if (raw == nullptr || *raw == '\0') {
        return default_value;
    }
    try {
        return std::stoi(raw);
    } catch (...) {
        return default_value;
    }
}

std::string env_str(const char* key, const char* default_value) {
    const char* raw = std::getenv(key);
    if (raw == nullptr || *raw == '\0') {
        return std::string(default_value);
    }
    return std::string(raw);
}

std::string trim(std::string s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) {
        s.erase(s.begin());
    }
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) {
        s.pop_back();
    }
    return s;
}

bool parse_bool(const std::string& s) {
    const std::string t = trim(s);
    return t == "1" || t == "true" || t == "True" || t == "TRUE";
}

std::vector<LinearCase> default_cases() {
    return {
        {"mlp_s", 32, 256, 256, false},
        {"mlp_s_bias", 32, 256, 256, true},
        {"mlp_m", 128, 1024, 1024, false},
        {"mlp_m_bias", 128, 1024, 1024, true},
        {"xfmr_flat_s", 1024, 768, 3072, false},
        {"xfmr_flat_s_bias", 1024, 768, 3072, true},
        {"xfmr_flat_m", 4096, 1024, 4096, false},
        {"xfmr_flat_m_bias", 4096, 1024, 4096, true},
        {"big_2k8k", 2000, 8000, 8000, false},
        {"big_2k8k_bias", 2000, 8000, 8000, true},
    };
}

std::vector<LinearCase> load_cases(const std::string& csv_path) {
    std::ifstream in(csv_path);
    if (!in.is_open()) {
        return default_cases();
    }

    std::vector<LinearCase> out;
    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        if (first) {
            first = false;
            if (line.find("name") != std::string::npos) {
                continue;
            }
        }

        std::stringstream ss(line);
        std::string name, m, k, n, bias;
        if (!std::getline(ss, name, ',')) {
            continue;
        }
        if (!std::getline(ss, m, ',')) {
            continue;
        }
        if (!std::getline(ss, k, ',')) {
            continue;
        }
        if (!std::getline(ss, n, ',')) {
            continue;
        }
        if (!std::getline(ss, bias, ',')) {
            continue;
        }

        LinearCase c{};
        c.name = trim(name);
        c.m = std::stoi(trim(m));
        c.k = std::stoi(trim(k));
        c.n = std::stoi(trim(n));
        c.bias = parse_bool(bias);
        if (c.m > 0 && c.k > 0 && c.n > 0) {
            out.push_back(c);
        }
    }

    if (out.empty()) {
        return default_cases();
    }
    return out;
}

bool cuda_available() {
    int device_count = 0;
    const cudaError_t err = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
}

__global__ void fill_linear_forward_backward_data(
    float* x,
    float* w,
    float* b,
    float* dy,
    size_t x_numel,
    size_t w_numel,
    size_t b_numel,
    size_t dy_numel) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < x_numel) {
        x[i] = static_cast<float>((i % 17) - 8) * 0.03125f;
    }
    if (i < w_numel) {
        w[i] = static_cast<float>((i % 23) - 11) * 0.015625f;
    }
    if (b != nullptr && i < b_numel) {
        b[i] = static_cast<float>((i % 7) - 3) * 0.0078125f;
    }
    if (i < dy_numel) {
        dy[i] = static_cast<float>((i % 29) - 14) * 0.015625f;
    }
}

double forward_backward_flops(const LinearCase* cfg) {
    const double m = static_cast<double>(cfg->m);
    const double n = static_cast<double>(cfg->n);
    const double k = static_cast<double>(cfg->k);

    const double forward_mm_flops = 2.0 * m * k * n;
    const double backward_mm_flops = 4.0 * m * n * k;
    const double forward_bias_flops = cfg->bias ? (m * n) : 0.0;
    const double backward_bias_flops = cfg->bias ? (m * n) : 0.0;
    return forward_mm_flops + backward_mm_flops + forward_bias_flops + backward_bias_flops;
}

void benchmark_linear_forward_backward(benchmark::State& state, const LinearCase* cfg) {
    if (!cuda_available()) {
        state.SkipWithError("CUDA device unavailable");
        return;
    }

    Stream stream;
    CublasHandle cublas_handle;

    // Keep X and W allocated on device for the benchmark lifetime so backward reuses
    // forward context pointers without host/device churn.
    Tensor x({cfg->m, cfg->k}, DType::F32, Device::CUDA, stream);
    Tensor w({cfg->n, cfg->k}, DType::F32, Device::CUDA, stream);
    Tensor b = cfg->bias ? Tensor({cfg->n}, DType::F32, Device::CUDA, stream)
                         : Tensor::empty({1}, DType::F32, Device::CUDA, stream);
    Tensor dy({cfg->m, cfg->n}, DType::F32, Device::CUDA, stream);

    const size_t x_numel = static_cast<size_t>(cfg->m) * static_cast<size_t>(cfg->k);
    const size_t w_numel = static_cast<size_t>(cfg->n) * static_cast<size_t>(cfg->k);
    const size_t b_numel = static_cast<size_t>(cfg->n);
    const size_t dy_numel = static_cast<size_t>(cfg->m) * static_cast<size_t>(cfg->n);
    const size_t max_numel = std::max(std::max(x_numel, w_numel), std::max(b_numel, dy_numel));
    const int threads = 256;
    const int blocks = static_cast<int>((max_numel + threads - 1) / threads);
    fill_linear_forward_backward_data<<<blocks, threads, 0, stream.s>>>(
        static_cast<float*>(x.data_),
        static_cast<float*>(w.data_),
        cfg->bias ? static_cast<float*>(b.data_) : nullptr,
        static_cast<float*>(dy.data_),
        x_numel,
        w_numel,
        b_numel,
        dy_numel);
    CUDA_CHECK(cudaGetLastError());
    stream.synchronize();

    const Tensor* b_ptr = cfg->bias ? &b : nullptr;
    const int warmup = env_int("FA_LINEAR_FB_WARMUP", 50);
    for (int i = 0; i < warmup; ++i) {
        LinearResults out = linear_forward(x, w, b_ptr, &stream, cublas_handle);
        (void)linear_backward(dy, out.ctx, true, true, cfg->bias, &stream, cublas_handle);
    }
    stream.synchronize();

    double total_seconds = 0.0;
    for (auto _ : state) {
        (void)_;
        Event start;
        Event stop;
        record(start, stream);
        LinearResults out = linear_forward(x, w, b_ptr, &stream, cublas_handle);
        LinearGrads grads = linear_backward(dy, out.ctx, true, true, cfg->bias, &stream, cublas_handle);
        (void)grads;
        record(stop, stream);
        stop.synchronize();
        const double seconds = static_cast<double>(elapsed_time(start, stop)) * 1.0e-3;
        total_seconds += seconds;
        state.SetIterationTime(seconds);
    }

    const double avg_seconds = total_seconds / static_cast<double>(state.iterations());
    const double flops = forward_backward_flops(cfg);
    const double tflops = (flops / avg_seconds) / 1.0e12;

    state.counters["M"] = static_cast<double>(cfg->m);
    state.counters["K"] = static_cast<double>(cfg->k);
    state.counters["N"] = static_cast<double>(cfg->n);
    state.counters["Bias"] = cfg->bias ? 1.0 : 0.0;
    state.counters["TFLOP/s"] = tflops;
}

}  // namespace

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);

    const std::string shapes_path = env_str("FA_LINEAR_FB_SHAPES", "bench/linear_shapes.csv");
    const int iters = env_int("FA_LINEAR_FB_ITERS", 200);
    const std::vector<LinearCase> cases = load_cases(shapes_path);

    if (cuda_available()) {
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        benchmark::AddCustomContext("gpu_name", prop.name);
    } else {
        benchmark::AddCustomContext("gpu_name", "unavailable");
    }
    benchmark::AddCustomContext("dtype", "float32");
    benchmark::AddCustomContext("shape_file", shapes_path);
    benchmark::AddCustomContext("iters", std::to_string(iters));
    benchmark::AddCustomContext("warmup", std::to_string(env_int("FA_LINEAR_FB_WARMUP", 50)));

    for (const LinearCase& c : cases) {
        const std::string bench_name =
            "LinearForwardBackward/name=" + c.name +
            "/m=" + std::to_string(c.m) +
            "/k=" + std::to_string(c.k) +
            "/n=" + std::to_string(c.n) +
            "/bias=" + std::to_string(c.bias ? 1 : 0) +
            "/mode=full";
        benchmark::RegisterBenchmark(bench_name.c_str(), benchmark_linear_forward_backward, &c)
            ->UseManualTime()
            ->Iterations(iters);
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
