#include "cu_stream.h"
#include "general.h"
#include "ops.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

std::vector<float> BuildRamp(size_t n, float start, float step) {
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = start + step * static_cast<float>(i);
    }
    return out;
}

TEST(OptimizerSgdUpdate, UpdatesInPlaceSmallTensor) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    Stream stream;
    Tensor param({2, 3}, DType::F32, Device::CUDA, stream);
    Tensor grad({2, 3}, DType::F32, Device::CUDA, stream);

    const std::vector<float> p0 = {1.0f, 2.0f, 3.0f, -1.0f, 0.5f, 2.5f};
    const std::vector<float> g0 = {0.1f, -0.2f, 0.4f, 1.0f, -1.5f, 2.0f};
    param.copy_from(p0, stream);
    grad.copy_from(g0, stream);

    sgd_update_(param, grad, 0.5f, stream);
    const std::vector<float> got = param.to_vector<float>(stream);

    ASSERT_EQ(got.size(), p0.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], p0[i] - 0.5f * g0[i], 1e-6f) << "idx=" << i;
    }
}

TEST(OptimizerSgdUpdate, UpdatesInPlaceGridStrideCoverage) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    Stream stream;
    constexpr int64_t kNumel = 4097;
    Tensor param({kNumel}, DType::F32, Device::CUDA, stream);
    Tensor grad({kNumel}, DType::F32, Device::CUDA, stream);

    const std::vector<float> p0 = BuildRamp(static_cast<size_t>(kNumel), -2.0f, 0.001f);
    const std::vector<float> g0 = BuildRamp(static_cast<size_t>(kNumel), 1.0f, -0.0005f);
    param.copy_from(p0, stream);
    grad.copy_from(g0, stream);

    sgd_update_(param, grad, 0.03f, stream);
    const std::vector<float> got = param.to_vector<float>(stream);

    ASSERT_EQ(got.size(), p0.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], p0[i] - 0.03f * g0[i], 1e-5f) << "idx=" << i;
    }
}

TEST(OptimizerSgdUpdate, RejectsInvalidInputs) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    Stream stream;
    Tensor p_f32({4}, DType::F32, Device::CUDA, stream);
    Tensor g_f32({4}, DType::F32, Device::CUDA, stream);
    Tensor g_f16({4}, DType::F16, Device::CUDA, stream);
    Tensor g_cpu({4}, DType::F32, Device::CPU);
    Tensor g_bad_shape({5}, DType::F32, Device::CUDA, stream);

    EXPECT_NO_THROW((void)sgd_update_(p_f32, g_f32, 0.1f));
    EXPECT_THROW((void)sgd_update_(p_f32, g_bad_shape, 0.1f, stream), std::invalid_argument);
    EXPECT_THROW((void)sgd_update_(p_f32, g_f16, 0.1f, stream), std::invalid_argument);
    EXPECT_THROW((void)sgd_update_(p_f32, g_cpu, 0.1f, stream), std::invalid_argument);
    EXPECT_THROW((void)sgd_update_(p_f32, g_f32, std::numeric_limits<float>::quiet_NaN(), stream), std::invalid_argument);
    EXPECT_THROW((void)sgd_update_(p_f32, g_f32, std::numeric_limits<float>::infinity(), stream), std::invalid_argument);
}

}  // namespace
