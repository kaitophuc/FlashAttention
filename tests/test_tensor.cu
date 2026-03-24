#include "allocator.h"
#include "cu_stream.h"
#include "general.h"
#include "tensor.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

TEST(TensorCorrectness, AllocFreeStress) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    Stream stream;
    std::mt19937_64 rng(0xC0FFEEULL);
    std::uniform_int_distribution<size_t> bytes_dist(1ULL << 10, 64ULL << 20);

    constexpr int kIters = 10000;
    constexpr int kWindow = 1000;
    std::vector<double> window_ms;
    window_ms.reserve(kIters / kWindow);

    auto window_start = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; ++i) {
        const size_t bytes = bytes_dist(rng);
        void* ptr = allocate_device(bytes, stream);
        deallocate_device(ptr, stream);

        if ((i + 1) % 256 == 0) {
            stream.synchronize();
        }
        if ((i + 1) % kWindow == 0) {
            stream.synchronize();
            const auto now = std::chrono::steady_clock::now();
            const double ms =
                std::chrono::duration<double, std::milli>(now - window_start).count();
            window_ms.push_back(ms);
            window_start = now;
        }
    }
    stream.synchronize();

    ASSERT_FALSE(window_ms.empty());
    const double first = window_ms.front();
    const double last = window_ms.back();
    ASSERT_GT(first, 0.0);
    ASSERT_GT(last, 0.0);

    // Basic anti-regression sanity: last window should not be drastically slower.
    const double slowdown = last / first;
    EXPECT_LT(slowdown, 4.0);
}

TEST(TensorCorrectness, H2DAndD2HRoundTrip) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    Stream stream(cudaStream_t(0));

    constexpr int64_t kN = 1LL << 20;  // 1M floats
    Tensor h({kN}, DType::F32, Device::CPU, stream);
    Tensor d({kN}, DType::F32, Device::CUDA, stream);
    Tensor h2({kN}, DType::F32, Device::CPU, stream);

    auto* h_ptr = static_cast<float*>(h.data_);
    auto* h2_ptr = static_cast<float*>(h2.data_);

    for (int64_t i = 0; i < kN; ++i) {
        h_ptr[i] = static_cast<float>(i) * 0.5f;
    }

    d.copy_from(h, stream);
    h2.copy_from(d, stream);
    stream.synchronize();

    constexpr float kEps = 1e-6f;
    for (int64_t i = 0; i < kN; ++i) {
        const float expected = static_cast<float>(i) * 0.5f;
        const float got = h2_ptr[i];
        EXPECT_NEAR(got, expected, kEps);
    }
}

}  // namespace
