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

    Stream stream;

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

    Stream stream;

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

    Stream stream;

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

    Stream stream;

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

    Stream stream;
    EXPECT_THROW((void)Tensor::zeros({128}, DType::F32, Device::CPU, stream), std::invalid_argument);
}

TEST(TensorCorrectness, RejectsCpuToVectorWithStreamArgument) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    Stream stream;
    Tensor h({128}, DType::F32, Device::CPU);
    EXPECT_THROW((void)h.to_vector<float>(stream), std::invalid_argument);
}

TEST(TensorCorrectness, RejectsCpuDestinationCopyFromWithStreamArgument) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    Stream stream;
    Tensor src({128}, DType::F32, Device::CPU);
    Tensor dst({128}, DType::F32, Device::CPU);
    EXPECT_THROW((void)dst.copy_from(src, stream), std::invalid_argument);
}

std::vector<float> ReferenceMatmulF32(const std::vector<float>& a,
                                      const std::vector<float>& b,
                                      int64_t m,
                                      int64_t k,
                                      int64_t n) {
    std::vector<float> out(static_cast<size_t>(m * n), 0.0f);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int64_t p = 0; p < k; ++p) {
                acc += a[static_cast<size_t>(i * k + p)] * b[static_cast<size_t>(p * n + j)];
            }
            out[static_cast<size_t>(i * n + j)] = acc;
        }
    }
    return out;
}

TEST(TensorMatmul, CpuF32MatchesReference) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    constexpr int64_t m = 2;
    constexpr int64_t k = 3;
    constexpr int64_t n = 4;
    const std::vector<float> a = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    const std::vector<float> b = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 1.0f, 2.0f, 3.0f
    };

    Tensor lhs({m, k}, DType::F32, Device::CPU);
    Tensor rhs({k, n}, DType::F32, Device::CPU);
    lhs.copy_from(a);
    rhs.copy_from(b);

    Tensor out = lhs.matmul(rhs);
    const std::vector<float> expected = ReferenceMatmulF32(a, b, m, k, n);
    const std::vector<float> got = out.to_vector<float>();

    ASSERT_EQ(got.size(), expected.size());
    constexpr float kEps = 1e-6f;
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(got[i], expected[i], kEps) << "idx=" << i;
    }
}

TEST(TensorMatmul, CpuI32MatchesReference) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    Tensor lhs({2, 2}, DType::I32, Device::CPU);
    Tensor rhs({2, 2}, DType::I32, Device::CPU);
    lhs.copy_from(std::vector<int32_t>{-1, 2, 3, -4});
    rhs.copy_from(std::vector<int32_t>{5, -6, 7, 8});

    Tensor out = lhs.matmul(rhs);
    const std::vector<int32_t> got = out.to_vector<int32_t>();
    const std::vector<int32_t> expected = {9, 22, -13, -50};

    ASSERT_EQ(got.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(got[i], expected[i]) << "idx=" << i;
    }
}

TEST(TensorMatmul, CpuRejectsInvalidInputs) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    Tensor a_1d({4}, DType::F32, Device::CPU);
    Tensor b_2d({4, 2}, DType::F32, Device::CPU);
    Tensor bad_k({5, 2}, DType::F32, Device::CPU);
    Tensor i32_rhs({4, 2}, DType::I32, Device::CPU);

    EXPECT_THROW((void)a_1d.matmul(b_2d), std::invalid_argument);
    EXPECT_THROW((void)b_2d.matmul(bad_k), std::invalid_argument);
    EXPECT_THROW((void)b_2d.matmul(i32_rhs), std::invalid_argument);
}

TEST(TensorMatmul, CpuApiUsesCurrentStreamForCudaMatmul) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    Stream stream;
    Tensor lhs({2, 3}, DType::F32, Device::CUDA, stream);
    Tensor rhs({3, 2}, DType::F32, Device::CUDA, stream);
    EXPECT_NO_THROW((void)lhs.matmul(rhs));
}

TEST(TensorMatmul, CudaF32MatchesReference) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    constexpr int64_t m = 3;
    constexpr int64_t k = 2;
    constexpr int64_t n = 4;
    const std::vector<float> a = {
        1.0f, 2.0f,
        -3.0f, 4.0f,
        5.5f, -6.5f
    };
    const std::vector<float> b = {
        7.0f, -8.0f, 9.0f, 1.0f,
        2.0f, 3.0f, -4.0f, 5.0f
    };

    Tensor lhs_h({m, k}, DType::F32, Device::CPU);
    Tensor rhs_h({k, n}, DType::F32, Device::CPU);
    lhs_h.copy_from(a);
    rhs_h.copy_from(b);

    Stream stream;
    CublasHandle handle;
    Tensor lhs_d = lhs_h.clone(Device::CUDA, stream);
    Tensor rhs_d = rhs_h.clone(Device::CUDA, stream);

    Tensor out_d = lhs_d.matmul(rhs_d, stream, handle);
    Tensor out_h = out_d.clone(Device::CPU);
    const std::vector<float> got = out_h.to_vector<float>();
    const std::vector<float> expected = ReferenceMatmulF32(a, b, m, k, n);

    ASSERT_EQ(got.size(), expected.size());
    constexpr float kEps = 1e-5f;
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(got[i], expected[i], kEps) << "idx=" << i;
    }
}

TEST(TensorMatmul, CudaF16MatchesReferenceForEdgeShape) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    // Edge shape: 1xK times Kx1.
    constexpr int64_t m = 1;
    constexpr int64_t k = 5;
    constexpr int64_t n = 1;
    const std::vector<float> a_f32 = {1.0f, -2.0f, 3.0f, -4.0f, 0.5f};
    const std::vector<float> b_f32 = {-0.25f, 0.5f, -1.0f, 2.0f, 4.0f};

    std::vector<__half> a_f16(a_f32.size());
    std::vector<__half> b_f16(b_f32.size());
    for (size_t i = 0; i < a_f32.size(); ++i) {
        a_f16[i] = __float2half(a_f32[i]);
    }
    for (size_t i = 0; i < b_f32.size(); ++i) {
        b_f16[i] = __float2half(b_f32[i]);
    }

    Tensor lhs_h({m, k}, DType::F16, Device::CPU);
    Tensor rhs_h({k, n}, DType::F16, Device::CPU);
    lhs_h.copy_from(a_f16);
    rhs_h.copy_from(b_f16);

    Stream stream;
    CublasHandle handle;
    Tensor lhs_d = lhs_h.clone(Device::CUDA, stream);
    Tensor rhs_d = rhs_h.clone(Device::CUDA, stream);

    Tensor out_d = lhs_d.matmul(rhs_d, stream, handle);
    Tensor out_h = out_d.clone(Device::CPU);
    const std::vector<__half> got_h = out_h.to_vector<__half>();
    ASSERT_EQ(got_h.size(), static_cast<size_t>(m * n));

    const std::vector<float> expected = ReferenceMatmulF32(a_f32, b_f32, m, k, n);
    constexpr float kEps = 2e-2f;
    EXPECT_NEAR(__half2float(got_h[0]), expected[0], kEps);
}

TEST(TensorMatmul, CudaRejectsInvalidInputs) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    Stream stream;
    CublasHandle handle;

    Tensor lhs({2, 3}, DType::F32, Device::CUDA, stream);
    Tensor rhs_bad_k({4, 2}, DType::F32, Device::CUDA, stream);
    Tensor rhs_i32({3, 2}, DType::I32, Device::CUDA, stream);
    Tensor rhs_cpu({3, 2}, DType::F32, Device::CPU);
    Tensor lhs_1d({6}, DType::F32, Device::CUDA, stream);

    EXPECT_THROW((void)lhs.matmul(rhs_bad_k, stream, handle), std::invalid_argument);
    EXPECT_THROW((void)lhs.matmul(rhs_i32, stream, handle), std::invalid_argument);
    EXPECT_THROW((void)lhs.matmul(rhs_cpu, stream, handle), std::invalid_argument);
    EXPECT_THROW((void)lhs_1d.matmul(lhs, stream, handle), std::invalid_argument);
}

TEST(TensorMatmul, CudaAcceptsNonDefaultStream) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    Stream stream;
    CublasHandle handle;
    Tensor lhs({2, 2}, DType::F32, Device::CUDA, stream);
    Tensor rhs({2, 2}, DType::F32, Device::CUDA, stream);

    cudaStream_t raw_non_default = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&raw_non_default, cudaStreamNonBlocking));
    Stream non_default_stream;
    non_default_stream.s = raw_non_default;
    non_default_stream.owns_ = false;

    EXPECT_NO_THROW((void)lhs.matmul(rhs, non_default_stream, handle));
    CUDA_CHECK(cudaStreamDestroy(raw_non_default));
}


}  // namespace
