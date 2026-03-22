#include "ops.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

bool cuda_available() {
    int device_count = 0;
    const cudaError_t err = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
}

std::vector<float> reference_linear(
    const std::vector<float>& x,
    const std::vector<float>& w,
    const std::vector<float>* b,
    int m,
    int k,
    int n) {
    std::vector<float> y(static_cast<size_t>(m) * static_cast<size_t>(n), 0.0f);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int t = 0; t < k; ++t) {
                acc += x[static_cast<size_t>(i) * k + t] *
                       w[static_cast<size_t>(j) * k + t];
            }
            if (b != nullptr) {
                acc += (*b)[j];
            }
            y[static_cast<size_t>(i) * n + j] = acc;
        }
    }
    return y;
}

TEST(LinearForward, RejectsNon2DX) {
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }
    Stream stream;
    CublasHandle handle;
    Tensor x({2, 3, 4}, DType::F32, Device::CUDA, stream);
    Tensor w({5, 4}, DType::F32, Device::CUDA, stream);

    EXPECT_THROW((void)linear_forward(x, w, nullptr, &stream, handle), std::invalid_argument);
}

TEST(LinearForward, RejectsKMismatch) {
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }
    Stream stream;
    CublasHandle handle;
    Tensor x({2, 3}, DType::F32, Device::CUDA, stream);
    Tensor w({4, 5}, DType::F32, Device::CUDA, stream);

    EXPECT_THROW((void)linear_forward(x, w, nullptr, &stream, handle), std::invalid_argument);
}

TEST(LinearForward, RejectsBiasShapeMismatch) {
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }
    Stream stream;
    CublasHandle handle;
    Tensor x({2, 3}, DType::F32, Device::CUDA, stream);
    Tensor w({4, 3}, DType::F32, Device::CUDA, stream);
    Tensor b_bad({5}, DType::F32, Device::CUDA, stream);

    EXPECT_THROW((void)linear_forward(x, w, &b_bad, &stream, handle), std::invalid_argument);
}

TEST(LinearForward, CUDAReferenceNoBias) {
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }
    Stream stream;
    CublasHandle handle;

    constexpr int m = 3;
    constexpr int k = 4;
    constexpr int n = 2;

    Tensor x_h({m, k}, DType::F32, Device::CPU, stream);
    Tensor w_h({n, k}, DType::F32, Device::CPU, stream);
    auto* x_h_ptr = static_cast<float*>(x_h.data_);
    auto* w_h_ptr = static_cast<float*>(w_h.data_);

    std::vector<float> x_ref(static_cast<size_t>(m) * k);
    std::vector<float> w_ref(static_cast<size_t>(n) * k);
    for (int i = 0; i < m * k; ++i) {
        const float v = static_cast<float>((i % 7) - 3) * 0.25f;
        x_h_ptr[i] = v;
        x_ref[i] = v;
    }
    for (int i = 0; i < n * k; ++i) {
        const float v = static_cast<float>((i % 5) - 2) * 0.5f;
        w_h_ptr[i] = v;
        w_ref[i] = v;
    }

    Tensor x_d = x_h.clone(Device::CUDA, stream);
    Tensor w_d = w_h.clone(Device::CUDA, stream);
    LinearResults out = linear_forward(x_d, w_d, nullptr, &stream, handle);
    Tensor y_h = out.Y.clone(Device::CPU, stream);
    stream.synchronize();

    const std::vector<float> expected = reference_linear(x_ref, w_ref, nullptr, m, k, n);
    auto* y_ptr = static_cast<float*>(y_h.data_);
    for (int i = 0; i < m * n; ++i) {
        EXPECT_NEAR(y_ptr[i], expected[static_cast<size_t>(i)], 1e-4f);
    }
}

TEST(LinearForward, CUDAReferenceWithBias) {
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }
    Stream stream;
    CublasHandle handle;

    constexpr int m = 2;
    constexpr int k = 3;
    constexpr int n = 4;

    Tensor x_h({m, k}, DType::F32, Device::CPU, stream);
    Tensor w_h({n, k}, DType::F32, Device::CPU, stream);
    Tensor b_h({n}, DType::F32, Device::CPU, stream);
    auto* x_h_ptr = static_cast<float*>(x_h.data_);
    auto* w_h_ptr = static_cast<float*>(w_h.data_);
    auto* b_h_ptr = static_cast<float*>(b_h.data_);

    std::vector<float> x_ref(static_cast<size_t>(m) * k);
    std::vector<float> w_ref(static_cast<size_t>(n) * k);
    std::vector<float> b_ref(static_cast<size_t>(n));
    for (int i = 0; i < m * k; ++i) {
        const float v = static_cast<float>(i + 1) * 0.1f;
        x_h_ptr[i] = v;
        x_ref[i] = v;
    }
    for (int i = 0; i < n * k; ++i) {
        const float v = static_cast<float>((i % 4) - 1) * 0.2f;
        w_h_ptr[i] = v;
        w_ref[i] = v;
    }
    for (int i = 0; i < n; ++i) {
        const float v = static_cast<float>(i - 2) * 0.05f;
        b_h_ptr[i] = v;
        b_ref[i] = v;
    }

    Tensor x_d = x_h.clone(Device::CUDA, stream);
    Tensor w_d = w_h.clone(Device::CUDA, stream);
    Tensor b_d = b_h.clone(Device::CUDA, stream);
    LinearResults out = linear_forward(x_d, w_d, &b_d, &stream, handle);
    Tensor y_h = out.Y.clone(Device::CPU, stream);
    stream.synchronize();

    const std::vector<float> expected = reference_linear(x_ref, w_ref, &b_ref, m, k, n);
    auto* y_ptr = static_cast<float*>(y_h.data_);
    for (int i = 0; i < m * n; ++i) {
        EXPECT_NEAR(y_ptr[i], expected[static_cast<size_t>(i)], 1e-4f);
    }
}

}  // namespace
