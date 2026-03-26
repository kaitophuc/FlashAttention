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
    Tensor h({kN}, DType::F32, Device::CPU);
    Tensor d({kN}, DType::F32, Device::CUDA, stream);
    Tensor h2({kN}, DType::F32, Device::CPU);

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

TEST(TensorCorrectness, CloneCudaToCpu_NoSegfaultAndMatches) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    Stream stream(cudaStream_t(0));

    constexpr int64_t kN = 1LL << 20;  // 1M floats
    Tensor h_src({kN}, DType::F32, Device::CPU);
    auto* h_src_ptr = static_cast<float*>(h_src.data_);
    for (int64_t i = 0; i < kN; ++i) {
        h_src_ptr[i] = static_cast<float>(i) * 0.25f - 17.0f;
    }

    Tensor d = h_src.clone(Device::CUDA, stream);
    stream.synchronize();

    // This is the path that previously crashed when clone(Device::CPU) used memcpy on device ptr.
    Tensor h_cloned = d.clone(Device::CPU);

    auto* h_cloned_ptr = static_cast<float*>(h_cloned.data_);
    constexpr float kEps = 1e-6f;
    for (int64_t i = 0; i < kN; ++i) {
        EXPECT_NEAR(h_cloned_ptr[i], h_src_ptr[i], kEps) << "idx=" << i;
    }
}

TEST(TensorCorrectness, CloneCpuToCudaToCpu_RoundTrip) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    Stream stream(cudaStream_t(0));

    constexpr int64_t kN = 1LL << 20;  // 1M floats
    Tensor h0({kN}, DType::F32, Device::CPU);
    auto* h0_ptr = static_cast<float*>(h0.data_);
    for (int64_t i = 0; i < kN; ++i) {
        h0_ptr[i] = std::sin(static_cast<float>(i) * 0.001f);
    }

    Tensor d = h0.clone(Device::CUDA, stream);
    stream.synchronize();

    Tensor h1 = d.clone(Device::CPU);

    auto* h1_ptr = static_cast<float*>(h1.data_);
    constexpr float kEps = 1e-6f;
    for (int64_t i = 0; i < kN; ++i) {
        EXPECT_NEAR(h1_ptr[i], h0_ptr[i], kEps) << "idx=" << i;
    }
}

TEST(TensorCorrectness, ToVectorFromCuda_IsSynchronized) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    Stream stream(cudaStream_t(0));

    constexpr int64_t kN = 1LL << 20;  // 1M floats
    Tensor h_src({kN}, DType::F32, Device::CPU);
    auto* h_src_ptr = static_cast<float*>(h_src.data_);
    for (int64_t i = 0; i < kN; ++i) {
        h_src_ptr[i] = static_cast<float>((i % 1024) - 512) * 0.03125f;
    }

    Tensor d({kN}, DType::F32, Device::CUDA, stream);
    d.copy_from(h_src, stream);
    stream.synchronize();

    // If to_vector() returns before D2H copy completion, this can read stale/garbage.
    const std::vector<float> v = d.to_vector<float>(stream);

    ASSERT_EQ(v.size(), static_cast<size_t>(kN));
    constexpr float kEps = 1e-6f;
    for (int64_t i = 0; i < kN; ++i) {
        EXPECT_NEAR(v[static_cast<size_t>(i)], h_src_ptr[i], kEps) << "idx=" << i;
    }
}

TEST(TensorCorrectness, RejectsCpuZerosWithStreamArgument) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    Stream stream(cudaStream_t(0));
    EXPECT_THROW((void)Tensor::zeros({128}, DType::F32, Device::CPU, stream), std::invalid_argument);
}

TEST(TensorCorrectness, RejectsCpuToVectorWithStreamArgument) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    Stream stream(cudaStream_t(0));
    Tensor h({128}, DType::F32, Device::CPU);
    EXPECT_THROW((void)h.to_vector<float>(stream), std::invalid_argument);
}

TEST(TensorCorrectness, RejectsCpuDestinationCopyFromWithStreamArgument) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    Stream stream(cudaStream_t(0));
    Tensor src({128}, DType::F32, Device::CPU);
    Tensor dst({128}, DType::F32, Device::CPU);
    EXPECT_THROW((void)dst.copy_from(src, stream), std::invalid_argument);
}


}  // namespace
