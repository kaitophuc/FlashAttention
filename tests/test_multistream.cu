#include "cu_stream.h"
#include "general.h"
#include "ops.h"
#include "tensor.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace {

void ExpectAllClose(const std::vector<float>& got,
                    const std::vector<float>& expected,
                    float atol = 1e-4f,
                    float rtol = 1e-4f) {
    ASSERT_EQ(got.size(), expected.size());
    for (size_t i = 0; i < got.size(); ++i) {
        const float tol = atol + rtol * std::fabs(expected[i]);
        EXPECT_LE(std::fabs(got[i] - expected[i]), tol) << "idx=" << i;
    }
}

Tensor MakeCudaTensorF32(const std::vector<int64_t>& shape,
                         const std::vector<float>& values,
                         Stream& stream) {
    Tensor t(shape, DType::F32, Device::CUDA, stream);
    t.copy_from(values, stream);
    stream.synchronize();
    return t;
}

Tensor MakeCudaTensorI32(const std::vector<int64_t>& shape,
                         const std::vector<int32_t>& values,
                         Stream& stream) {
    Tensor t(shape, DType::I32, Device::CUDA, stream);
    t.copy_from(values, stream);
    stream.synchronize();
    return t;
}

TEST(MultiStream, LinearInterleavedTwoStreams) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream a = stream_from_pool(4);
    Stream b = stream_from_pool(5);
    CublasHandle& handle = current_cublas_handle();

    Tensor x1 = MakeCudaTensorF32({2, 2}, {1.0f, 2.0f, -1.0f, 3.0f}, a);
    Tensor w1 = MakeCudaTensorF32({2, 2}, {2.0f, 0.0f, 1.0f, -1.0f}, a);
    Tensor b1 = MakeCudaTensorF32({2}, {0.5f, -0.5f}, a);

    Tensor x2 = MakeCudaTensorF32({2, 2}, {2.0f, -1.0f, 4.0f, 0.5f}, b);
    Tensor w2 = MakeCudaTensorF32({2, 2}, {1.5f, 2.0f, -3.0f, 1.0f}, b);
    Tensor b2 = MakeCudaTensorF32({2}, {1.0f, -2.0f}, b);

    LinearResults y1 = linear_forward(x1, w1, &b1, a, handle);
    LinearResults y2 = linear_forward(x2, w2, &b2, b, handle);

    a.synchronize();
    b.synchronize();

    ExpectAllClose(y1.Y.to_vector<float>(a), {2.5f, -1.5f, -1.5f, -4.5f});
    ExpectAllClose(y2.Y.to_vector<float>(b), {2.0f, -9.0f, 8.0f, -13.5f});
}

TEST(MultiStream, LayerNormInterleavedTwoStreams) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream a = stream_from_pool(6);
    Stream b = stream_from_pool(7);

    Tensor x1 = MakeCudaTensorF32({2, 4}, {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.0f, 1.0f, 2.0f}, a);
    Tensor g1 = MakeCudaTensorF32({4}, {1.0f, 1.0f, 1.0f, 1.0f}, a);
    Tensor bt1 = MakeCudaTensorF32({4}, {0.0f, 0.0f, 0.0f, 0.0f}, a);

    Tensor x2 = MakeCudaTensorF32({2, 4}, {2.0f, 1.0f, 0.0f, -1.0f, 3.0f, -3.0f, 1.0f, -1.0f}, b);
    Tensor g2 = MakeCudaTensorF32({4}, {1.0f, 1.0f, 1.0f, 1.0f}, b);
    Tensor bt2 = MakeCudaTensorF32({4}, {0.0f, 0.0f, 0.0f, 0.0f}, b);

    LayerNormResults y1 = layernorm_forward(x1, g1, bt1, 1e-5f, a);
    LayerNormResults y2 = layernorm_forward(x2, g2, bt2, 1e-5f, b);

    a.synchronize();
    b.synchronize();

    const std::vector<float> v1 = y1.Y.to_vector<float>(a);
    const std::vector<float> v2 = y2.Y.to_vector<float>(b);

    ASSERT_EQ(v1.size(), static_cast<size_t>(8));
    ASSERT_EQ(v2.size(), static_cast<size_t>(8));

    for (int row = 0; row < 2; ++row) {
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        for (int col = 0; col < 4; ++col) {
            sum1 += v1[row * 4 + col];
            sum2 += v2[row * 4 + col];
        }
        EXPECT_NEAR(sum1, 0.0f, 1e-3f);
        EXPECT_NEAR(sum2, 0.0f, 1e-3f);
    }
}

TEST(MultiStream, ReluInterleavedTwoStreams) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream a = stream_from_pool(8);
    Stream b = stream_from_pool(9);

    Tensor x1 = MakeCudaTensorF32({2, 3}, {-1.0f, 0.0f, 2.0f, -3.0f, 4.0f, -5.0f}, a);
    Tensor x2 = MakeCudaTensorF32({2, 3}, {1.0f, -2.0f, 3.0f, -4.0f, 0.0f, 5.0f}, b);

    ReluResults y1 = relu_forward(x1, a);
    ReluResults y2 = relu_forward(x2, b);

    a.synchronize();
    b.synchronize();

    ExpectAllClose(y1.Y.to_vector<float>(a), {0.0f, 0.0f, 2.0f, 0.0f, 4.0f, 0.0f});
    ExpectAllClose(y2.Y.to_vector<float>(b), {1.0f, 0.0f, 3.0f, 0.0f, 0.0f, 5.0f});
}

TEST(MultiStream, SoftmaxAndBackwardInterleavedTwoStreams) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream a = stream_from_pool(10);
    Stream b = stream_from_pool(11);

    Tensor x1 = MakeCudaTensorF32({2, 3}, {1.0f, 2.0f, 3.0f, -1.0f, 0.0f, 1.0f}, a);
    Tensor x2 = MakeCudaTensorF32({2, 3}, {2.0f, 0.0f, -1.0f, 4.0f, 2.0f, 0.0f}, b);

    Tensor y1 = softmax_forward(x1, a);
    Tensor y2 = softmax_forward(x2, b);

    Tensor dy1 = MakeCudaTensorF32({2, 3}, {1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -1.0f}, a);
    Tensor dy2 = MakeCudaTensorF32({2, 3}, {-1.0f, 0.25f, 0.75f, 0.0f, -2.0f, 2.0f}, b);

    SoftmaxGrads dx1 = softmax_backward(dy1, y1, a);
    SoftmaxGrads dx2 = softmax_backward(dy2, y2, b);

    a.synchronize();
    b.synchronize();

    const std::vector<float> y1v = y1.to_vector<float>(a);
    const std::vector<float> y2v = y2.to_vector<float>(b);
    const std::vector<float> dx1v = dx1.dX.to_vector<float>(a);
    const std::vector<float> dx2v = dx2.dX.to_vector<float>(b);

    for (int row = 0; row < 2; ++row) {
        float sumy1 = 0.0f;
        float sumy2 = 0.0f;
        float sumdx1 = 0.0f;
        float sumdx2 = 0.0f;
        for (int col = 0; col < 3; ++col) {
            sumy1 += y1v[row * 3 + col];
            sumy2 += y2v[row * 3 + col];
            sumdx1 += dx1v[row * 3 + col];
            sumdx2 += dx2v[row * 3 + col];
        }
        EXPECT_NEAR(sumy1, 1.0f, 1e-5f);
        EXPECT_NEAR(sumy2, 1.0f, 1e-5f);
        EXPECT_NEAR(sumdx1, 0.0f, 1e-4f);
        EXPECT_NEAR(sumdx2, 0.0f, 1e-4f);
    }
}

TEST(MultiStream, SoftmaxCrossEntropyInterleavedTwoStreams) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream a = stream_from_pool(12);
    Stream b = stream_from_pool(13);

    Tensor logits1 = MakeCudaTensorF32({2, 3}, {2.0f, 1.0f, 0.0f, -1.0f, 0.5f, 3.0f}, a);
    Tensor labels1 = MakeCudaTensorI32({2}, {0, 2}, a);

    Tensor logits2 = MakeCudaTensorF32({2, 3}, {1.0f, -1.0f, 0.0f, 0.5f, 2.0f, -0.5f}, b);
    Tensor labels2 = MakeCudaTensorI32({2}, {2, 1}, b);

    SoftmaxCrossEntropyResults out1 = softmax_cross_entropy_forward(logits1, labels1, a);
    SoftmaxCrossEntropyResults out2 = softmax_cross_entropy_forward(logits2, labels2, b);

    SoftmaxCrossEntropyGrads g1 = softmax_cross_entropy_backward(out1.ctx, a);
    SoftmaxCrossEntropyGrads g2 = softmax_cross_entropy_backward(out2.ctx, b);

    a.synchronize();
    b.synchronize();

    const std::vector<float> loss1 = out1.loss.to_vector<float>(a);
    const std::vector<float> loss2 = out2.loss.to_vector<float>(b);
    ASSERT_EQ(loss1.size(), static_cast<size_t>(1));
    ASSERT_EQ(loss2.size(), static_cast<size_t>(1));
    EXPECT_TRUE(std::isfinite(loss1[0]));
    EXPECT_TRUE(std::isfinite(loss2[0]));
    EXPECT_GT(loss1[0], 0.0f);
    EXPECT_GT(loss2[0], 0.0f);

    const std::vector<float> dx1 = g1.dX.to_vector<float>(a);
    const std::vector<float> dx2 = g2.dX.to_vector<float>(b);

    for (int row = 0; row < 2; ++row) {
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        for (int col = 0; col < 3; ++col) {
            sum1 += dx1[row * 3 + col];
            sum2 += dx2[row * 3 + col];
        }
        EXPECT_NEAR(sum1, 0.0f, 1e-4f);
        EXPECT_NEAR(sum2, 0.0f, 1e-4f);
    }
}

TEST(MultiStream, ClassificationAndOptimizerInterleavedTwoStreams) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream a = stream_from_pool(14);
    Stream b = stream_from_pool(15);

    Tensor logits = MakeCudaTensorF32(
        {4, 3},
        {
            2.0f, 1.0f, 0.0f,
            0.1f, 0.7f, 0.2f,
            -1.0f, 0.0f, 1.0f,
            0.2f, 0.9f, 0.3f,
        },
        a);
    Tensor labels = MakeCudaTensorI32({4}, {0, 1, 0, 1}, a);

    Tensor param = MakeCudaTensorF32({2, 3}, {1.0f, -2.0f, 3.5f, 4.0f, 0.0f, -1.5f}, b);
    Tensor grad = MakeCudaTensorF32({2, 3}, {0.2f, -0.1f, 1.0f, -2.0f, 3.0f, 0.5f}, b);

    Tensor correct = classification_correct_count(logits, labels, a);
    sgd_update_(param, grad, 0.25f, b);

    a.synchronize();
    b.synchronize();

    const std::vector<int32_t> got_correct = correct.to_vector<int32_t>(a);
    ASSERT_EQ(got_correct.size(), static_cast<size_t>(1));
    EXPECT_EQ(got_correct[0], 3);

    ExpectAllClose(param.to_vector<float>(b), {0.95f, -1.975f, 3.25f, 4.5f, -0.75f, -1.625f});
}

TEST(MultiStream, LinearForwardOnABackwardOnBWithEventDependency) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream a = stream_from_pool(16);
    Stream b = stream_from_pool(17);
    CublasHandle& handle = current_cublas_handle();

    Tensor x = MakeCudaTensorF32({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, a);
    Tensor w = MakeCudaTensorF32({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f}, a);
    Tensor dY = MakeCudaTensorF32({2, 2}, {1.0f, 1.0f, 1.0f, 1.0f}, b);

    LinearResults out = linear_forward(x, w, nullptr, a, handle);

    Event fwd_done;
    record(fwd_done, a);
    wait(b, fwd_done);

    LinearGrads grads = linear_backward(dY, out.ctx, true, true, false, b, handle);
    b.synchronize();

    ASSERT_TRUE(grads.has_dX);
    ASSERT_TRUE(grads.has_dW);
    ASSERT_FALSE(grads.has_db);

    ExpectAllClose(grads.dX.value().to_vector<float>(b), {1.0f, 1.0f, 1.0f, 1.0f});
    ExpectAllClose(grads.dW.value().to_vector<float>(b), {4.0f, 6.0f, 4.0f, 6.0f});
}

TEST(MultiStreamStress, LinearAndMatmulInterleavingReusesThreadLocalCublasHandle) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream a = stream_from_pool(18);
    Stream b = stream_from_pool(19);
    CublasHandle& handle = current_cublas_handle();

    Tensor lx = MakeCudaTensorF32({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, a);
    Tensor lw = MakeCudaTensorF32({2, 2}, {2.0f, 1.0f, 0.0f, -1.0f}, a);

    Tensor mx = MakeCudaTensorF32({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f}, a);
    Tensor mw = MakeCudaTensorF32({2, 2}, {4.0f, 2.0f, 3.0f, 1.0f}, a);

    std::optional<Tensor> last_linear_a;
    std::optional<Tensor> last_linear_b;
    std::optional<Tensor> last_matmul_a;
    std::optional<Tensor> last_matmul_b;

    for (int i = 0; i < 400; ++i) {
        Stream& s = (i % 2 == 0) ? a : b;
        LinearResults l = linear_forward(lx, lw, nullptr, s, handle);
        Tensor m = mx.matmul(mw, s, handle);

        if (i % 2 == 0) {
            last_linear_a.emplace(std::move(l.Y));
            last_matmul_a.emplace(std::move(m));
        } else {
            last_linear_b.emplace(std::move(l.Y));
            last_matmul_b.emplace(std::move(m));
        }
    }

    a.synchronize();
    b.synchronize();

    ASSERT_TRUE(last_linear_a.has_value());
    ASSERT_TRUE(last_linear_b.has_value());
    ASSERT_TRUE(last_matmul_a.has_value());
    ASSERT_TRUE(last_matmul_b.has_value());

    const std::vector<float> expected_linear = {4.0f, -2.0f, 10.0f, -4.0f};
    const std::vector<float> expected_matmul = {4.0f, 2.0f, 3.0f, 1.0f};

    ExpectAllClose(last_linear_a.value().to_vector<float>(a), expected_linear);
    ExpectAllClose(last_linear_b.value().to_vector<float>(b), expected_linear);
    ExpectAllClose(last_matmul_a.value().to_vector<float>(a), expected_matmul);
    ExpectAllClose(last_matmul_b.value().to_vector<float>(b), expected_matmul);
}

TEST(MultiStreamStress, AllocatorLifetimeAcrossStreamSwitchesRemainsStable) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    const int pool_n = stream_pool_size();
    Stream original = current_stream();

    for (int i = 0; i < 300; ++i) {
        Stream sa = stream_from_pool(i % pool_n);
        Stream sb = stream_from_pool((i + 11) % pool_n);

        std::vector<float> payload(256, static_cast<float>((i % 13) - 5));

        set_current_stream(sa);
        auto owned = std::make_unique<Tensor>(std::vector<int64_t>{256}, DType::F32, Device::CUDA, sa);
        owned->copy_from(payload, sa);

        Event written;
        record(written, sa);
        wait(sb, written);

        set_current_stream(sb);
        owned.reset();

        Tensor probe({256}, DType::F32, Device::CUDA, sb);
        probe.copy_from(payload, sb);

        if (i % 50 == 0) {
            const std::vector<float> got = probe.to_vector<float>(sb);
            EXPECT_NEAR(got[17], payload[17], 1e-6f);
        }
    }

    set_current_stream(original);
}

}  // namespace
