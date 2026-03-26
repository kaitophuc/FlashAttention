#include "general.h"
#include "test_linear.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string ReproTag(const std::string& name, uint32_t seed, int m, int n, int k) {
    std::ostringstream os;
    os << "case=" << name << " seed=" << seed << " m=" << m << " n=" << n << " k=" << k;
    return os.str();
}

void ExpectVectorNear(const std::vector<float>& got,
                      const std::vector<float>& expected,
                      float abs_tol,
                      float rel_tol,
                      const std::string& ctx) {
    ASSERT_EQ(got.size(), expected.size()) << ctx;
    for (size_t i = 0; i < got.size(); ++i) {
        const float tol = abs_tol + rel_tol * std::fabs(expected[i]);
        EXPECT_NEAR(got[i], expected[i], tol) << ctx << " idx=" << i;
    }
}

Tensor MakeCpuTensor2D(int rows, int cols, const std::vector<float>& value) {
    Tensor t({rows, cols}, DType::F32, Device::CPU);
    t.copy_from(value);
    return t;
}

Tensor MakeCpuTensor1D(int n, const std::vector<float>& values) {
    Tensor t({n}, DType::F32, Device::CPU);
    t.copy_from(values);
    return t;
}

std::vector<float> RunForwardToHost(const Tensor& x_h,
                                    const Tensor& w_h,
                                    const Tensor* b_h,
                                    Stream& stream,
                                    CublasHandle& handle) {
    Tensor x_d = x_h.clone(Device::CUDA, stream);
    Tensor w_d = w_h.clone(Device::CUDA, stream);

    std::optional<Tensor> b_d;
    const Tensor* b_d_ptr = nullptr;
    if (b_h != nullptr) {
        b_d.emplace(b_h->clone(Device::CUDA, stream));
        b_d_ptr = &b_d.value();
    }

    LinearResults out = linear_forward(x_d, w_d, b_d_ptr, &stream, handle);
    Tensor y_h = out.Y.clone(Device::CPU);
    stream.synchronize();

    return y_h.to_vector<float>();
}

float Dot(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Dot inputs must have the same size.");
    }
    double acc = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        acc += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return static_cast<float>(acc);
}

float ForwardLossDotDY(const std::vector<float>& x,
                       const std::vector<float>& w,
                       const std::vector<float>* b,
                       const std::vector<float>& dY_ref,
                       int m,
                       int n,
                       int k,
                       Stream& stream,
                       CublasHandle& handle) {
    Tensor x_h = MakeCpuTensor2D(m, k, x);
    Tensor w_h = MakeCpuTensor2D(n, k, w);

    std::optional<Tensor> b_h;
    const Tensor* b_h_ptr = nullptr;
    if (b != nullptr) {
        b_h.emplace(MakeCpuTensor1D(n, *b));
        b_h_ptr = &b_h.value();
    }

    const std::vector<float> y = RunForwardToHost(x_h, w_h, b_h_ptr, stream, handle);
    return Dot(y, dY_ref);
}

TEST(LinearBackward, MatchesReferenceAllGradsOddN) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 13;
    constexpr int n = 37;
    constexpr int k = 19;
    constexpr float abs_tol = 1e-4f;
    constexpr float rel_tol = 1e-4f;

    const uint32_t seed = fa_test::MixSeed(fa_test::kLinearSeedBase, m, n, k, 101);
    fa_test::HostLinearInputs<float> inputs(m, n, k, true, -1.0f, 1.0f, seed);
    ASSERT_TRUE(inputs.b_h.has_value());

    const std::vector<float> dY_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 17u);

    Stream stream;
    CublasHandle handle;
    Tensor x_d = inputs.x_h.clone(Device::CUDA, stream);
    Tensor w_d = inputs.w_h.clone(Device::CUDA, stream);
    Tensor b_d = inputs.b_h->clone(Device::CUDA, stream);

    LinearResults out = linear_forward(x_d, w_d, &b_d, &stream, handle);

    Tensor dY_h = MakeCpuTensor2D(m, n, dY_ref);
    Tensor dY_d = dY_h.clone(Device::CUDA, stream);

    LinearGrads grads = linear_backward(dY_d, out.ctx, true, true, true, &stream, handle);
    ASSERT_TRUE(grads.has_dX);
    ASSERT_TRUE(grads.has_dW);
    ASSERT_TRUE(grads.has_db);
    ASSERT_TRUE(grads.dX.has_value());
    ASSERT_TRUE(grads.dW.has_value());
    ASSERT_TRUE(grads.db.has_value());

    Tensor dX_h = grads.dX->clone(Device::CPU);
    Tensor dW_h = grads.dW->clone(Device::CPU);
    Tensor db_h = grads.db->clone(Device::CPU);
    stream.synchronize();

    std::vector<float> dX_expected, dW_expected, db_expected;
    fa_test::reference_linear_backward(inputs.x_ref, inputs.w_ref, dY_ref, m, k, n,
                                       dX_expected, dW_expected, db_expected);

    ExpectVectorNear(dX_h.to_vector<float>(), dX_expected, abs_tol, rel_tol,
                     ReproTag("backward_dx", seed, m, n, k));
    ExpectVectorNear(dW_h.to_vector<float>(), dW_expected, abs_tol, rel_tol,
                     ReproTag("backward_dw", seed, m, n, k));
    ExpectVectorNear(db_h.to_vector<float>(), db_expected, abs_tol, rel_tol,
                     ReproTag("backward_db", seed, m, n, k));
}

TEST(LinearBackward, NeedsDbIgnoredWhenForwardHadNoBias) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 9;
    constexpr int n = 11;
    constexpr int k = 7;

    const uint32_t seed = fa_test::MixSeed(fa_test::kLinearSeedBase, m, n, k, 102);
    fa_test::HostLinearInputs<float> inputs(m, n, k, false, -1.0f, 1.0f, seed);

    const std::vector<float> dY_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -0.5f, 0.5f, seed + 23u);

    Stream stream;
    CublasHandle handle;
    Tensor x_d = inputs.x_h.clone(Device::CUDA, stream);
    Tensor w_d = inputs.w_h.clone(Device::CUDA, stream);
    LinearResults out = linear_forward(x_d, w_d, nullptr, &stream, handle);

    Tensor dY_h = MakeCpuTensor2D(m, n, dY_ref);
    Tensor dY_d = dY_h.clone(Device::CUDA, stream);

    LinearGrads grads = linear_backward(dY_d, out.ctx, false, false, true, &stream, handle);
    EXPECT_FALSE(grads.has_dX);
    EXPECT_FALSE(grads.has_dW);
    EXPECT_FALSE(grads.has_db);
    EXPECT_FALSE(grads.dX.has_value());
    EXPECT_FALSE(grads.dW.has_value());
    EXPECT_FALSE(grads.db.has_value());
}

TEST(LinearBackward, RejectsDYShapeMismatch) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream stream;
    CublasHandle handle;

    Tensor x({5, 4}, DType::F32, Device::CUDA, stream);
    Tensor w({3, 4}, DType::F32, Device::CUDA, stream);
    LinearResults out = linear_forward(x, w, nullptr, &stream, handle);

    Tensor dY_bad({6, 3}, DType::F32, Device::CUDA, stream);
    EXPECT_THROW((void)linear_backward(dY_bad, out.ctx, true, true, false, &stream, handle),
                 std::invalid_argument);
}

TEST(LinearBackward, RejectsNullStream) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream stream;
    CublasHandle handle;
    Tensor x({5, 4}, DType::F32, Device::CUDA, stream);
    Tensor w({3, 4}, DType::F32, Device::CUDA, stream);
    LinearResults out = linear_forward(x, w, nullptr, &stream, handle);
    Tensor dY({5, 3}, DType::F32, Device::CUDA, stream);

    EXPECT_THROW((void)linear_backward(dY, out.ctx, true, true, false, nullptr, handle),
                 std::invalid_argument);
}

TEST(LinearBackward, RejectsNonDefaultStream) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream stream;
    CublasHandle handle;
    Tensor x({5, 4}, DType::F32, Device::CUDA, stream);
    Tensor w({3, 4}, DType::F32, Device::CUDA, stream);
    LinearResults out = linear_forward(x, w, nullptr, &stream, handle);
    Tensor dY({5, 3}, DType::F32, Device::CUDA, stream);

    cudaStream_t raw_non_default = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&raw_non_default, cudaStreamNonBlocking));
    Stream non_default_stream;
    non_default_stream.s = raw_non_default;
    non_default_stream.owns_ = false;

    EXPECT_THROW((void)linear_backward(dY, out.ctx, true, true, false, &non_default_stream, handle),
                 std::invalid_argument);

    CUDA_CHECK(cudaStreamDestroy(raw_non_default));
}

TEST(LinearBackward, RejectsNullCtxPointers) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream stream;
    CublasHandle handle;
    Tensor dY({5, 3}, DType::F32, Device::CUDA, stream);
    const LinearCtx bad_ctx{nullptr, nullptr, false, 5, 3, 4};

    EXPECT_THROW((void)linear_backward(dY, bad_ctx, true, true, false, &stream, handle),
                 std::invalid_argument);
}

TEST(LinearBackward, RejectsNon2DDY) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream stream;
    CublasHandle handle;
    Tensor x({5, 4}, DType::F32, Device::CUDA, stream);
    Tensor w({3, 4}, DType::F32, Device::CUDA, stream);
    const LinearCtx ctx{&x, &w, false, 5, 3, 4};
    Tensor dY_bad({5, 3, 2}, DType::F32, Device::CUDA, stream);

    EXPECT_THROW((void)linear_backward(dY_bad, ctx, true, true, false, &stream, handle),
                 std::invalid_argument);
}

TEST(LinearBackward, RejectsDYDTypeNotF32) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream stream;
    CublasHandle handle;
    Tensor x({5, 4}, DType::F32, Device::CUDA, stream);
    Tensor w({3, 4}, DType::F32, Device::CUDA, stream);
    const LinearCtx ctx{&x, &w, false, 5, 3, 4};
    Tensor dY_bad({5, 3}, DType::F16, Device::CUDA, stream);

    EXPECT_THROW((void)linear_backward(dY_bad, ctx, true, true, false, &stream, handle),
                 std::invalid_argument);
}

TEST(LinearBackward, RejectsDYDTypeMismatchCtx) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream stream;
    CublasHandle handle;
    Tensor x({5, 4}, DType::F16, Device::CUDA, stream);
    Tensor w({3, 4}, DType::F16, Device::CUDA, stream);
    const LinearCtx ctx{&x, &w, false, 5, 3, 4};
    Tensor dY({5, 3}, DType::F32, Device::CUDA, stream);

    EXPECT_THROW((void)linear_backward(dY, ctx, true, true, false, &stream, handle),
                 std::invalid_argument);
}

TEST(LinearBackward, RejectsDYDeviceMismatchCtx) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream stream;
    CublasHandle handle;
    Tensor x({5, 4}, DType::F32, Device::CUDA, stream);
    Tensor w({3, 4}, DType::F32, Device::CUDA, stream);
    const LinearCtx ctx{&x, &w, false, 5, 3, 4};
    Tensor dY_cpu({5, 3}, DType::F32, Device::CPU);

    EXPECT_THROW((void)linear_backward(dY_cpu, ctx, true, true, false, &stream, handle),
                 std::invalid_argument);
}

TEST(LinearBackward, RejectsInvalidKInCtx) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream stream;
    CublasHandle handle;
    Tensor x({5, 4}, DType::F32, Device::CUDA, stream);
    Tensor w({3, 4}, DType::F32, Device::CUDA, stream);
    const LinearCtx bad_ctx{&x, &w, false, 5, 3, 0};
    Tensor dY({5, 3}, DType::F32, Device::CUDA, stream);

    EXPECT_THROW((void)linear_backward(dY, bad_ctx, true, true, false, &stream, handle),
                 std::invalid_argument);
}

TEST(LinearBackward, NeedsDbFalseSkipsBiasGradEvenWithBias) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 7;
    constexpr int n = 9;
    constexpr int k = 5;
    const uint32_t seed = fa_test::MixSeed(fa_test::kLinearSeedBase, m, n, k, 213);

    fa_test::HostLinearInputs<float> inputs(m, n, k, true, -1.0f, 1.0f, seed);
    ASSERT_TRUE(inputs.b_h.has_value());
    const std::vector<float> dY_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 59u);

    Stream stream;
    CublasHandle handle;
    Tensor x_d = inputs.x_h.clone(Device::CUDA, stream);
    Tensor w_d = inputs.w_h.clone(Device::CUDA, stream);
    Tensor b_d = inputs.b_h->clone(Device::CUDA, stream);
    LinearResults out = linear_forward(x_d, w_d, &b_d, &stream, handle);

    Tensor dY_h = MakeCpuTensor2D(m, n, dY_ref);
    Tensor dY_d = dY_h.clone(Device::CUDA, stream);
    LinearGrads grads = linear_backward(dY_d, out.ctx, true, true, false, &stream, handle);

    EXPECT_TRUE(grads.has_dX);
    EXPECT_TRUE(grads.has_dW);
    EXPECT_FALSE(grads.has_db);
    ASSERT_TRUE(grads.dX.has_value());
    ASSERT_TRUE(grads.dW.has_value());
    EXPECT_FALSE(grads.db.has_value());
}

TEST(LinearBackward, NeedsGradFlagsAllFalseReturnsNoGrads) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 7;
    constexpr int n = 9;
    constexpr int k = 5;
    const uint32_t seed = fa_test::MixSeed(fa_test::kLinearSeedBase, m, n, k, 214);

    fa_test::HostLinearInputs<float> inputs(m, n, k, true, -1.0f, 1.0f, seed);
    ASSERT_TRUE(inputs.b_h.has_value());
    const std::vector<float> dY_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 61u);

    Stream stream;
    CublasHandle handle;
    Tensor x_d = inputs.x_h.clone(Device::CUDA, stream);
    Tensor w_d = inputs.w_h.clone(Device::CUDA, stream);
    Tensor b_d = inputs.b_h->clone(Device::CUDA, stream);
    LinearResults out = linear_forward(x_d, w_d, &b_d, &stream, handle);

    Tensor dY_h = MakeCpuTensor2D(m, n, dY_ref);
    Tensor dY_d = dY_h.clone(Device::CUDA, stream);
    LinearGrads grads = linear_backward(dY_d, out.ctx, false, false, false, &stream, handle);

    EXPECT_FALSE(grads.has_dX);
    EXPECT_FALSE(grads.has_dW);
    EXPECT_FALSE(grads.has_db);
    EXPECT_FALSE(grads.dX.has_value());
    EXPECT_FALSE(grads.dW.has_value());
    EXPECT_FALSE(grads.db.has_value());
}

TEST(LinearBackward, NeedsGradFlagsDXOnlyMatchesReference) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 8;
    constexpr int n = 6;
    constexpr int k = 5;
    constexpr float abs_tol = 1e-4f;
    constexpr float rel_tol = 1e-4f;
    const uint32_t seed = fa_test::MixSeed(fa_test::kLinearSeedBase, m, n, k, 215);

    fa_test::HostLinearInputs<float> inputs(m, n, k, true, -1.0f, 1.0f, seed);
    ASSERT_TRUE(inputs.b_h.has_value());
    const std::vector<float> dY_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 71u);

    Stream stream;
    CublasHandle handle;
    Tensor x_d = inputs.x_h.clone(Device::CUDA, stream);
    Tensor w_d = inputs.w_h.clone(Device::CUDA, stream);
    Tensor b_d = inputs.b_h->clone(Device::CUDA, stream);
    LinearResults out = linear_forward(x_d, w_d, &b_d, &stream, handle);

    Tensor dY_h = MakeCpuTensor2D(m, n, dY_ref);
    Tensor dY_d = dY_h.clone(Device::CUDA, stream);
    LinearGrads grads = linear_backward(dY_d, out.ctx, true, false, false, &stream, handle);

    EXPECT_TRUE(grads.has_dX);
    EXPECT_FALSE(grads.has_dW);
    EXPECT_FALSE(grads.has_db);
    ASSERT_TRUE(grads.dX.has_value());
    EXPECT_FALSE(grads.dW.has_value());
    EXPECT_FALSE(grads.db.has_value());

    std::vector<float> dX_expected, dW_expected, db_expected;
    fa_test::reference_linear_backward(inputs.x_ref, inputs.w_ref, dY_ref, m, k, n,
                                       dX_expected, dW_expected, db_expected);
    stream.synchronize();
    const std::vector<float> dX = grads.dX->clone(Device::CPU).to_vector<float>();
    ExpectVectorNear(dX, dX_expected, abs_tol, rel_tol, ReproTag("flags_dx_only", seed, m, n, k));
}

TEST(LinearBackward, SweepAllCases) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    const std::vector<fa_test::LinearBackwardCase> cases = fa_test::BuildBackwardCases();
    std::vector<std::string> failures;
    failures.reserve(64);

    auto check_vec = [](const std::vector<float>& got,
                        const std::vector<float>& expected,
                        float abs_tol,
                        float rel_tol) {
        int fail_count = 0;
        float worst_abs_err = 0.0f;
        float worst_tol = 0.0f;
        int worst_idx = -1;
        if (got.size() != expected.size()) {
            return std::tuple<int, float, float, int>{-1, 0.0f, 0.0f, -1};
        }
        for (size_t i = 0; i < got.size(); ++i) {
            const float e = expected[i];
            const float abs_err = std::fabs(got[i] - e);
            const float tol = abs_tol + rel_tol * std::fabs(e);
            if (abs_err > tol) {
                ++fail_count;
                if (abs_err > worst_abs_err) {
                    worst_abs_err = abs_err;
                    worst_tol = tol;
                    worst_idx = static_cast<int>(i);
                }
            }
        }
        return std::tuple<int, float, float, int>{fail_count, worst_abs_err, worst_tol, worst_idx};
    };

    for (const fa_test::LinearBackwardCase& c : cases) {
        for (int iter = 0; iter < c.iters; ++iter) {
            const uint32_t seed = fa_test::MixSeed(fa_test::kLinearSeedBase, c.m, c.n, c.k, iter + 300);
            fa_test::HostLinearInputs<float> inputs(c.m, c.n, c.k, c.with_bias, c.lo, c.hi, seed);
            const std::vector<float> dY_ref = fa_test::SampleUniformVector(
                static_cast<size_t>(c.m) * c.n, c.lo, c.hi, seed + 7u);

            Stream stream;
            CublasHandle handle;

            Tensor x_d = inputs.x_h.clone(Device::CUDA, stream);
            Tensor w_d = inputs.w_h.clone(Device::CUDA, stream);
            std::optional<Tensor> b_d;
            const Tensor* b_d_ptr = nullptr;
            if (c.with_bias) {
                ASSERT_TRUE(inputs.b_h.has_value());
                b_d.emplace(inputs.b_h->clone(Device::CUDA, stream));
                b_d_ptr = &b_d.value();
            }
            LinearResults out = linear_forward(x_d, w_d, b_d_ptr, &stream, handle);

            Tensor dY_h = MakeCpuTensor2D(c.m, c.n, dY_ref);
            Tensor dY_d = dY_h.clone(Device::CUDA, stream);
            LinearGrads grads = linear_backward(dY_d, out.ctx, true, true, c.with_bias, &stream, handle);
            ASSERT_TRUE(grads.dX.has_value());
            ASSERT_TRUE(grads.dW.has_value());
            if (c.with_bias) {
                ASSERT_TRUE(grads.db.has_value());
            } else {
                ASSERT_FALSE(grads.db.has_value());
            }

            stream.synchronize();
            const std::vector<float> dX_got = grads.dX->clone(Device::CPU).to_vector<float>();
            const std::vector<float> dW_got = grads.dW->clone(Device::CPU).to_vector<float>();
            std::vector<float> db_got;
            if (c.with_bias) {
                db_got = grads.db->clone(Device::CPU).to_vector<float>();
            }

            std::vector<float> dX_expected, dW_expected, db_expected;
            fa_test::reference_linear_backward(inputs.x_ref, inputs.w_ref, dY_ref, c.m, c.k, c.n,
                                               dX_expected, dW_expected, db_expected);

            const auto [dx_fail, dx_worst, dx_tol, dx_idx] =
                check_vec(dX_got, dX_expected, c.abs_tol, c.rel_tol);
            const auto [dw_fail, dw_worst, dw_tol, dw_idx] =
                check_vec(dW_got, dW_expected, c.abs_tol, c.rel_tol);
            int db_fail = 0;
            float db_worst = 0.0f;
            float db_tol = 0.0f;
            int db_idx = -1;
            if (c.with_bias) {
                std::tie(db_fail, db_worst, db_tol, db_idx) =
                    check_vec(db_got, db_expected, c.abs_tol, c.rel_tol);
            }

            const bool has_failure = (dx_fail != 0) || (dw_fail != 0) || (c.with_bias && db_fail != 0);
            if (has_failure) {
                std::ostringstream one;
                one << "case=" << c.name
                    << " dist=" << fa_test::DistName(c.dist)
                    << " bias=" << (c.with_bias ? "true" : "false")
                    << " iter=" << iter
                    << " seed=" << seed
                    << " m=" << c.m << " n=" << c.n << " k=" << c.k
                    << " dX_fail=" << dx_fail << " dX_worst_idx=" << dx_idx
                    << " dX_worst_abs_err=" << dx_worst << " dX_worst_tol=" << dx_tol
                    << " dW_fail=" << dw_fail << " dW_worst_idx=" << dw_idx
                    << " dW_worst_abs_err=" << dw_worst << " dW_worst_tol=" << dw_tol;
                if (c.with_bias) {
                    one << " db_fail=" << db_fail << " db_worst_idx=" << db_idx
                        << " db_worst_abs_err=" << db_worst << " db_worst_tol=" << db_tol;
                }
                failures.push_back(one.str());
            }
        }
    }

    if (!failures.empty()) {
        std::ostringstream all;
        all << "LinearBackward sweep failed in " << failures.size() << " case(s):\n";
        for (const std::string& f : failures) {
            all << "  " << f << "\n";
        }
        ADD_FAILURE() << all.str();
    }
}

TEST(LinearBackward, FiniteDifferenceGradientCheckSmallWithBias) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 3;
    constexpr int n = 4;
    constexpr int k = 2;
    constexpr float eps = 1e-3f;
    constexpr float abs_tol = 3e-2f;
    constexpr float rel_tol = 2e-2f;

    const uint32_t seed = fa_test::MixSeed(fa_test::kLinearSeedBase, m, n, k, 211);
    fa_test::HostLinearInputs<float> inputs(m, n, k, true, -0.6f, 0.6f, seed);
    ASSERT_TRUE(inputs.b_h.has_value());
    const std::vector<float>& x = inputs.x_ref;
    const std::vector<float>& w = inputs.w_ref;
    const std::vector<float>& b = inputs.b_ref;
    const std::vector<float> dY_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -0.6f, 0.6f, seed + 1u);

    Stream stream;
    CublasHandle handle;

    Tensor x_d = inputs.x_h.clone(Device::CUDA, stream);
    Tensor w_d = inputs.w_h.clone(Device::CUDA, stream);
    Tensor b_d = inputs.b_h->clone(Device::CUDA, stream);

    LinearResults out = linear_forward(x_d, w_d, &b_d, &stream, handle);

    Tensor dY_h = MakeCpuTensor2D(m, n, dY_ref);
    Tensor dY_d = dY_h.clone(Device::CUDA, stream);
    LinearGrads grads = linear_backward(dY_d, out.ctx, true, true, true, &stream, handle);
    ASSERT_TRUE(grads.dX.has_value());
    ASSERT_TRUE(grads.dW.has_value());
    ASSERT_TRUE(grads.db.has_value());
    stream.synchronize();

    const std::vector<float> dX = grads.dX->clone(Device::CPU).to_vector<float>();
    const std::vector<float> dW = grads.dW->clone(Device::CPU).to_vector<float>();
    const std::vector<float> db = grads.db->clone(Device::CPU).to_vector<float>();

    for (size_t i = 0; i < x.size(); ++i) {
        std::vector<float> x_plus = x;
        std::vector<float> x_minus = x;
        x_plus[i] += eps;
        x_minus[i] -= eps;
        const float lp = ForwardLossDotDY(x_plus, w, &b, dY_ref, m, n, k, stream, handle);
        const float lm = ForwardLossDotDY(x_minus, w, &b, dY_ref, m, n, k, stream, handle);
        const float g_num = (lp - lm) / (2.0f * eps);
        const float tol = abs_tol + rel_tol * std::fabs(g_num);
        EXPECT_NEAR(dX[i], g_num, tol) << ReproTag("fd_dx", seed, m, n, k) << " idx=" << i;
    }

    for (size_t i = 0; i < w.size(); ++i) {
        std::vector<float> w_plus = w;
        std::vector<float> w_minus = w;
        w_plus[i] += eps;
        w_minus[i] -= eps;
        const float lp = ForwardLossDotDY(x, w_plus, &b, dY_ref, m, n, k, stream, handle);
        const float lm = ForwardLossDotDY(x, w_minus, &b, dY_ref, m, n, k, stream, handle);
        const float g_num = (lp - lm) / (2.0f * eps);
        const float tol = abs_tol + rel_tol * std::fabs(g_num);
        EXPECT_NEAR(dW[i], g_num, tol) << ReproTag("fd_dw", seed, m, n, k) << " idx=" << i;
    }

    for (size_t i = 0; i < b.size(); ++i) {
        std::vector<float> b_plus = b;
        std::vector<float> b_minus = b;
        b_plus[i] += eps;
        b_minus[i] -= eps;
        const float lp = ForwardLossDotDY(x, w, &b_plus, dY_ref, m, n, k, stream, handle);
        const float lm = ForwardLossDotDY(x, w, &b_minus, dY_ref, m, n, k, stream, handle);
        const float g_num = (lp - lm) / (2.0f * eps);
        const float tol = abs_tol + rel_tol * std::fabs(g_num);
        EXPECT_NEAR(db[i], g_num, tol) << ReproTag("fd_db", seed, m, n, k) << " idx=" << i;
    }
}

TEST(LinearBackward, NeedsGradFlagsSelectiveOutputsAndValues) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 6;
    constexpr int n = 5;
    constexpr int k = 4;
    constexpr float abs_tol = 1e-4f;
    constexpr float rel_tol = 1e-4f;
    const uint32_t seed = fa_test::MixSeed(fa_test::kLinearSeedBase, m, n, k, 212);

    fa_test::HostLinearInputs<float> inputs(m, n, k, true, -1.0f, 1.0f, seed);
    ASSERT_TRUE(inputs.b_h.has_value());

    const std::vector<float> dY_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 31u);

    Stream stream;
    CublasHandle handle;
    Tensor x_d = inputs.x_h.clone(Device::CUDA, stream);
    Tensor w_d = inputs.w_h.clone(Device::CUDA, stream);
    Tensor b_d = inputs.b_h->clone(Device::CUDA, stream);
    LinearResults out = linear_forward(x_d, w_d, &b_d, &stream, handle);

    Tensor dY_h = MakeCpuTensor2D(m, n, dY_ref);
    Tensor dY_d = dY_h.clone(Device::CUDA, stream);
    LinearGrads grads = linear_backward(dY_d, out.ctx, false, true, true, &stream, handle);

    EXPECT_FALSE(grads.has_dX);
    EXPECT_TRUE(grads.has_dW);
    EXPECT_TRUE(grads.has_db);
    EXPECT_FALSE(grads.dX.has_value());
    ASSERT_TRUE(grads.dW.has_value());
    ASSERT_TRUE(grads.db.has_value());

    std::vector<float> dX_expected, dW_expected, db_expected;
    fa_test::reference_linear_backward(inputs.x_ref, inputs.w_ref, dY_ref, m, k, n,
                                       dX_expected, dW_expected, db_expected);

    stream.synchronize();
    const std::vector<float> dW = grads.dW->clone(Device::CPU).to_vector<float>();
    const std::vector<float> db = grads.db->clone(Device::CPU).to_vector<float>();

    ExpectVectorNear(dW, dW_expected, abs_tol, rel_tol, ReproTag("flags_dw", seed, m, n, k));
    ExpectVectorNear(db, db_expected, abs_tol, rel_tol, ReproTag("flags_db", seed, m, n, k));
}

TEST(LinearBackward, SingleStreamOrderingReuseStressFixedShape) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    const int launches = fa_test::LongStressEnabled() ? 900 : 350;
    constexpr int m = 96;
    constexpr int n = 144;
    constexpr int k = 80;
    constexpr float abs_tol = 2e-4f;
    constexpr float rel_tol = 2e-4f;

    Stream stream;
    CublasHandle handle;
    std::vector<fa_test::QueuedLinearBackwardJob> jobs;
    jobs.reserve(static_cast<size_t>(launches));

    for (int i = 0; i < launches; ++i) {
        const uint32_t seed = fa_test::MixSeed(fa_test::kLinearSeedBase, m, n, k, i);
        const bool with_bias = fa_test::DeterministicBool(seed);
        fa_test::HostLinearInputs<float> inputs(m, n, k, with_bias, -1.0f, 1.0f, seed);

        const std::vector<float> dY_ref =
            fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 37u);

        Tensor x_d = inputs.x_h.clone(Device::CUDA, stream);
        Tensor w_d = inputs.w_h.clone(Device::CUDA, stream);
        std::optional<Tensor> b_d;
        const Tensor* b_d_ptr = nullptr;
        if (with_bias) {
            ASSERT_TRUE(inputs.b_h.has_value());
            b_d.emplace(inputs.b_h->clone(Device::CUDA, stream));
            b_d_ptr = &b_d.value();
        }

        LinearResults out = linear_forward(x_d, w_d, b_d_ptr, &stream, handle);
        Tensor dY_h = MakeCpuTensor2D(m, n, dY_ref);
        Tensor dY_d = dY_h.clone(Device::CUDA, stream);

        LinearGrads grads = linear_backward(dY_d, out.ctx, true, true, with_bias, &stream, handle);
        ASSERT_TRUE(grads.dX.has_value());
        ASSERT_TRUE(grads.dW.has_value());
        if (with_bias) {
            ASSERT_TRUE(grads.db.has_value());
        } else {
            ASSERT_FALSE(grads.db.has_value());
        }

        std::vector<float> dX_expected, dW_expected, db_expected;
        fa_test::reference_linear_backward(inputs.x_ref, inputs.w_ref, dY_ref, m, k, n,
                                           dX_expected, dW_expected, db_expected);

        jobs.push_back(fa_test::QueuedLinearBackwardJob{
            m,
            n,
            k,
            with_bias,
            abs_tol,
            rel_tol,
            std::move(x_d),
            std::move(w_d),
            std::move(b_d),
            std::move(dY_d),
            grads.dX->clone(Device::CPU),
            grads.dW->clone(Device::CPU),
            with_bias ? std::optional<Tensor>(grads.db->clone(Device::CPU)) : std::nullopt,
            std::move(dX_expected),
            std::move(dW_expected),
            std::move(db_expected),
        });
    }

    stream.synchronize();
    fa_test::ValidateQueuedBackwardJobs(jobs);
}

TEST(LinearBackward, SingleStreamOrderingReuseStressShapeCycleABC) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    const fa_test::BackwardShapeCfg cfgs[3] = {
        {17, 29, 13, -1.0f, 1.0f, 1e-4f, 1e-4f, false},
        {73, 41, 59, -1e-3f, 1e-3f, 1e-6f, 1e-4f, true},
        {128, 96, 64, -10.0f, 10.0f, 4e-3f, 4e-4f, true},
    };

    const int launches = fa_test::LongStressEnabled() ? 600 : 240;
    Stream stream;
    CublasHandle handle;
    std::vector<fa_test::QueuedLinearBackwardJob> jobs;
    jobs.reserve(static_cast<size_t>(launches));

    for (int i = 0; i < launches; ++i) {
        const fa_test::BackwardShapeCfg& c = cfgs[i % 3];
        const uint32_t seed = fa_test::MixSeed(fa_test::kLinearSeedBase, c.m, c.n, c.k, i);
        fa_test::HostLinearInputs<float> inputs(c.m, c.n, c.k, c.with_bias, c.lo, c.hi, seed);

        const std::vector<float> dY_ref =
            fa_test::SampleUniformVector(static_cast<size_t>(c.m) * c.n, c.lo, c.hi, seed + 43u);

        Tensor x_d = inputs.x_h.clone(Device::CUDA, stream);
        Tensor w_d = inputs.w_h.clone(Device::CUDA, stream);
        std::optional<Tensor> b_d;
        const Tensor* b_d_ptr = nullptr;
        if (c.with_bias) {
            ASSERT_TRUE(inputs.b_h.has_value());
            b_d.emplace(inputs.b_h->clone(Device::CUDA, stream));
            b_d_ptr = &b_d.value();
        }

        LinearResults out = linear_forward(x_d, w_d, b_d_ptr, &stream, handle);
        Tensor dY_h = MakeCpuTensor2D(c.m, c.n, dY_ref);
        Tensor dY_d = dY_h.clone(Device::CUDA, stream);

        LinearGrads grads = linear_backward(dY_d, out.ctx, true, true, c.with_bias, &stream, handle);
        ASSERT_TRUE(grads.dX.has_value());
        ASSERT_TRUE(grads.dW.has_value());
        if (c.with_bias) {
            ASSERT_TRUE(grads.db.has_value());
        } else {
            ASSERT_FALSE(grads.db.has_value());
        }

        std::vector<float> dX_expected, dW_expected, db_expected;
        fa_test::reference_linear_backward(inputs.x_ref, inputs.w_ref, dY_ref, c.m, c.k, c.n,
                                           dX_expected, dW_expected, db_expected);

        jobs.push_back(fa_test::QueuedLinearBackwardJob{
            c.m,
            c.n,
            c.k,
            c.with_bias,
            c.abs_tol,
            c.rel_tol,
            std::move(x_d),
            std::move(w_d),
            std::move(b_d),
            std::move(dY_d),
            grads.dX->clone(Device::CPU),
            grads.dW->clone(Device::CPU),
            c.with_bias ? std::optional<Tensor>(grads.db->clone(Device::CPU)) : std::nullopt,
            std::move(dX_expected),
            std::move(dW_expected),
            std::move(db_expected),
        });
    }

    stream.synchronize();
    fa_test::ValidateQueuedBackwardJobs(jobs);
}

TEST(LinearForward, RejectsNon2DX) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }
    Stream stream;
    CublasHandle handle;
    Tensor x({20, 30, 40}, DType::F32, Device::CUDA, stream);
    Tensor w({50, 40}, DType::F32, Device::CUDA, stream);

    EXPECT_THROW((void)linear_forward(x, w, nullptr, &stream, handle), std::invalid_argument);
}

TEST(LinearForward, RejectsNon2DW) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }
    Stream stream;
    CublasHandle handle;
    Tensor x({20, 30}, DType::F32, Device::CUDA, stream);
    Tensor w({40, 30, 2}, DType::F32, Device::CUDA, stream);

    EXPECT_THROW((void)linear_forward(x, w, nullptr, &stream, handle), std::invalid_argument);
}

TEST(LinearForward, RejectsKMismatch) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }
    Stream stream;
    CublasHandle handle;
    Tensor x({20, 30}, DType::F32, Device::CUDA, stream);
    Tensor w({40, 50}, DType::F32, Device::CUDA, stream);

    EXPECT_THROW((void)linear_forward(x, w, nullptr, &stream, handle), std::invalid_argument);
}

TEST(LinearForward, RejectsBiasShapeMismatch) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }
    Stream stream;
    CublasHandle handle;
    Tensor x({20, 30}, DType::F32, Device::CUDA, stream);
    Tensor w({40, 30}, DType::F32, Device::CUDA, stream);
    Tensor b_bad({50}, DType::F32, Device::CUDA, stream);

    EXPECT_THROW((void)linear_forward(x, w, &b_bad, &stream, handle), std::invalid_argument);
}

TEST(LinearForward, RejectsNon1DBias) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }
    Stream stream;
    CublasHandle handle;
    Tensor x({20, 30}, DType::F32, Device::CUDA, stream);
    Tensor w({40, 30}, DType::F32, Device::CUDA, stream);
    Tensor b_bad({40, 1}, DType::F32, Device::CUDA, stream);

    EXPECT_THROW((void)linear_forward(x, w, &b_bad, &stream, handle), std::invalid_argument);
}

TEST(LinearForward, RejectsDTypeMismatchXW) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }
    Stream stream;
    CublasHandle handle;
    Tensor x({20, 30}, DType::F32, Device::CUDA, stream);
    Tensor w({40, 30}, DType::F16, Device::CUDA, stream);

    EXPECT_THROW((void)linear_forward(x, w, nullptr, &stream, handle), std::invalid_argument);
}

TEST(LinearForward, RejectsDTypeMismatchBias) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }
    Stream stream;
    CublasHandle handle;
    Tensor x({20, 30}, DType::F32, Device::CUDA, stream);
    Tensor w({40, 30}, DType::F32, Device::CUDA, stream);
    Tensor b_bad({40}, DType::F16, Device::CUDA, stream);

    EXPECT_THROW((void)linear_forward(x, w, &b_bad, &stream, handle), std::invalid_argument);
}

TEST(LinearForward, RejectsDeviceMismatchXW) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }
    Stream stream;
    CublasHandle handle;
    Tensor x({20, 30}, DType::F32, Device::CUDA, stream);
    Tensor w({40, 30}, DType::F32, Device::CPU);

    EXPECT_THROW((void)linear_forward(x, w, nullptr, &stream, handle), std::invalid_argument);
}

TEST(LinearForward, RejectsDeviceMismatchBias) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }
    Stream stream;
    CublasHandle handle;
    Tensor x({20, 30}, DType::F32, Device::CUDA, stream);
    Tensor w({40, 30}, DType::F32, Device::CUDA, stream);
    Tensor b_bad({40}, DType::F32, Device::CPU);

    EXPECT_THROW((void)linear_forward(x, w, &b_bad, &stream, handle), std::invalid_argument);
}

TEST(LinearForward, RejectsCPUOnlyInputsForNow) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }
    Stream stream;
    CublasHandle handle;
    Tensor x({20, 30}, DType::F32, Device::CPU);
    Tensor w({40, 30}, DType::F32, Device::CPU);

    EXPECT_THROW((void)linear_forward(x, w, nullptr, &stream, handle), std::invalid_argument);
}

TEST(LinearForward, RejectsNullStream) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }
    Stream stream;
    CublasHandle handle;
    Tensor x({20, 30}, DType::F32, Device::CUDA, stream);
    Tensor w({40, 30}, DType::F32, Device::CUDA, stream);

    EXPECT_THROW((void)linear_forward(x, w, nullptr, nullptr, handle), std::invalid_argument);
}

TEST(LinearForward, RejectsNonDefaultStream) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }
    Stream stream;
    CublasHandle handle;
    Tensor x({20, 30}, DType::F32, Device::CUDA, stream);
    Tensor w({40, 30}, DType::F32, Device::CUDA, stream);

    cudaStream_t raw_non_default = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&raw_non_default, cudaStreamNonBlocking));

    Stream non_default_stream;
    non_default_stream.s = raw_non_default;
    non_default_stream.owns_ = false;
    EXPECT_THROW((void)linear_forward(x, w, nullptr, &non_default_stream, handle), std::invalid_argument);

    CUDA_CHECK(cudaStreamDestroy(raw_non_default));
}

TEST(LinearForward, AcceptsDefaultStream) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }
    Stream stream;
    CublasHandle handle;
    Tensor x({4, 8}, DType::F32, Device::CUDA, stream);
    Tensor w({16, 8}, DType::F32, Device::CUDA, stream);

    EXPECT_NO_THROW((void)linear_forward(x, w, nullptr, &stream, handle));
}

TEST(LinearForward, SweepAllCases) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    const std::vector<fa_test::LinearCase> cases = fa_test::BuildCases();
    std::vector<std::string> failures;
    failures.reserve(32);

    for (const fa_test::LinearCase& c : cases) {
        for (int iter = 0; iter < c.iters; ++iter) {
            const uint32_t seed = fa_test::MixSeed(fa_test::kLinearSeedBase, c.m, c.n, c.k, iter);
            fa_test::HostLinearInputs<float> inputs(c.m, c.n, c.k, c.with_bias, c.lo, c.hi, seed);

            Stream stream;
            CublasHandle handle;

            Tensor x_d = inputs.x_h.clone(Device::CUDA, stream);
            Tensor w_d = inputs.w_h.clone(Device::CUDA, stream);

            std::optional<Tensor> b_d;
            const Tensor* b_d_ptr = nullptr;
            if (c.with_bias) {
                ASSERT_TRUE(inputs.b_h.has_value());
                b_d.emplace(inputs.b_h->clone(Device::CUDA, stream));
                b_d_ptr = &b_d.value();
            }

            LinearResults out = linear_forward(x_d, w_d, b_d_ptr, &stream, handle);
            Tensor y_h = out.Y.clone(Device::CPU);
            stream.synchronize();

            const std::vector<float>* b_ref = c.with_bias ? &inputs.b_ref : nullptr;
            const std::vector<float> expected =
                fa_test::reference_linear(inputs.x_ref, inputs.w_ref, b_ref, c.m, c.k, c.n);

            int fail_count = 0;
            float worst_abs_err = 0.0f;
            float worst_tol = 0.0f;
            int worst_idx = -1;

            auto* y_ptr = static_cast<float*>(y_h.data_);
            for (int i = 0; i < c.m * c.n; ++i) {
                const float e = expected[static_cast<size_t>(i)];
                const float got = y_ptr[i];
                const float abs_err = std::fabs(got - e);
                const float tol = c.abs_tol + c.rel_tol * std::fabs(e);

                if (abs_err > tol) {
                    ++fail_count;
                    if (abs_err > worst_abs_err) {
                        worst_abs_err = abs_err;
                        worst_tol = tol;
                        worst_idx = i;
                    }
                }
            }

            if (fail_count > 0) {
                std::ostringstream one;
                one << "case=" << c.name
                    << " dist=" << fa_test::DistName(c.dist)
                    << " bias=" << (c.with_bias ? "true" : "false")
                    << " iter=" << iter
                    << " seed=" << seed
                    << " m=" << c.m << " n=" << c.n << " k=" << c.k
                    << " fail_count=" << fail_count
                    << " worst_idx=" << worst_idx
                    << " worst_abs_err=" << worst_abs_err
                    << " worst_tol=" << worst_tol;
                failures.push_back(one.str());
            }
        }
    }

    if (!failures.empty()) {
        std::ostringstream all;
        all << "LinearForward sweep failed in " << failures.size() << " case(s):\n";
        for (const std::string& f : failures) {
            all << "  " << f << "\n";
        }
        ADD_FAILURE() << all.str();
    }
}

TEST(LinearForward, NumericEdgePatternsAndShapes) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    std::vector<fa_test::ForwardEdgeCase> cases;

    // Degenerate all-zero case (m=n=k=1).
    cases.push_back(fa_test::ForwardEdgeCase{"zeros_111", 1, 1, 1, true, 1e-7f, 1e-7f, {0.0f}, {0.0f}, {0.0f}});

    // One-hot/identity-like behavior.
    {
        const int m = 4;
        const int n = 4;
        const int k = 4;
        std::vector<float> x(static_cast<size_t>(m) * k, 0.0f);
        std::vector<float> w(static_cast<size_t>(n) * k, 0.0f);
        for (int i = 0; i < 4; ++i) {
            x[static_cast<size_t>(i) * k + i] = 1.0f;
            w[static_cast<size_t>(i) * k + i] = 1.0f;
        }
        std::vector<float> b = {0.5f, -0.5f, 0.25f, -0.25f};
        cases.push_back(fa_test::ForwardEdgeCase{"one_hot_identity", m, n, k, true, 1e-6f, 1e-6f, x, w, b});
    }

    // Adversarial near-cancellation/mixed magnitude.
    {
        const int m = 3;
        const int n = 5;
        const int k = 7;
        std::vector<float> x(static_cast<size_t>(m) * k);
        std::vector<float> w(static_cast<size_t>(n) * k);
        for (int i = 0; i < m * k; ++i) {
            const float big = (i % 2 == 0) ? 1.0e3f : -1.0e3f;
            const float tiny = (i % 3 == 0) ? 1.0e-3f : -1.0e-3f;
            x[static_cast<size_t>(i)] = big + tiny;
        }
        for (int i = 0; i < n * k; ++i) {
            const float sign = (i % 2 == 0) ? 1.0f : -1.0f;
            const float scale = (i % 5 == 0) ? 1.0e-3f : 1.0f;
            w[static_cast<size_t>(i)] = sign * scale;
        }
        std::vector<float> b(static_cast<size_t>(n), 0.0f);
        cases.push_back(fa_test::ForwardEdgeCase{"mixed_mag_cancellation", m, n, k, true, 2.0e-2f, 2.0e-4f, x, w, b});
    }

    // Tall-skinny random.
    {
        const int m = 256;
        const int n = 8;
        const int k = 16;
        const uint32_t seed = fa_test::MixSeed(fa_test::kLinearSeedBase, m, n, k, 1);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> x(static_cast<size_t>(m) * k);
        std::vector<float> w(static_cast<size_t>(n) * k);
        std::vector<float> b(static_cast<size_t>(n));
        for (float& v : x) v = dist(rng);
        for (float& v : w) v = dist(rng);
        for (float& v : b) v = dist(rng);
        cases.push_back(fa_test::ForwardEdgeCase{"tall_skinny", m, n, k, true, 1e-4f, 1e-4f, x, w, b});
    }

    // Short-fat random.
    {
        const int m = 8;
        const int n = 256;
        const int k = 16;
        const uint32_t seed = fa_test::MixSeed(fa_test::kLinearSeedBase, m, n, k, 2);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> x(static_cast<size_t>(m) * k);
        std::vector<float> w(static_cast<size_t>(n) * k);
        std::vector<float> b(static_cast<size_t>(n));
        for (float& v : x) v = dist(rng);
        for (float& v : w) v = dist(rng);
        for (float& v : b) v = dist(rng);
        cases.push_back(fa_test::ForwardEdgeCase{"short_fat", m, n, k, true, 1e-4f, 1e-4f, x, w, b});
    }

    for (const fa_test::ForwardEdgeCase& c : cases) {
        Stream stream;
        CublasHandle handle;

        Tensor x_h = MakeCpuTensor2D(c.m, c.k, c.x);
        Tensor w_h = MakeCpuTensor2D(c.n, c.k, c.w);
        std::optional<Tensor> b_h;
        const Tensor* b_h_ptr = nullptr;
        const std::vector<float>* b_ref = nullptr;
        if (c.with_bias) {
            b_h.emplace(MakeCpuTensor1D(c.n, c.b));
            b_h_ptr = &b_h.value();
            b_ref = &c.b;
        }

        const std::vector<float> got = RunForwardToHost(x_h, w_h, b_h_ptr, stream, handle);
        const std::vector<float> expected = fa_test::reference_linear(c.x, c.w, b_ref, c.m, c.k, c.n);

        const uint32_t seed = fa_test::MixSeed(fa_test::kLinearSeedBase, c.m, c.n, c.k);
        ExpectVectorNear(got, expected, c.abs_tol, c.rel_tol, ReproTag(c.name, seed, c.m, c.n, c.k));
    }
}

TEST(LinearForward, InvariantLinearityInInputNoBias) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    constexpr int m = 23;
    constexpr int n = 31;
    constexpr int k = 17;
    constexpr float abs_tol = 2e-4f;
    constexpr float rel_tol = 2e-4f;

    const uint32_t seed = fa_test::MixSeed(fa_test::kLinearSeedBase, m, n, k, 7);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<float> x1(static_cast<size_t>(m) * k);
    std::vector<float> x2(static_cast<size_t>(m) * k);
    std::vector<float> x12(static_cast<size_t>(m) * k);
    std::vector<float> w(static_cast<size_t>(n) * k);

    for (size_t i = 0; i < x1.size(); ++i) {
        x1[i] = dist(rng);
        x2[i] = dist(rng);
        x12[i] = x1[i] + x2[i];
    }
    for (float& v : w) v = dist(rng);

    Stream stream;
    CublasHandle handle;
    Tensor x1_h = MakeCpuTensor2D(m, k, x1);
    Tensor x2_h = MakeCpuTensor2D(m, k, x2);
    Tensor x12_h = MakeCpuTensor2D(m, k, x12);
    Tensor w_h = MakeCpuTensor2D(n, k, w);

    const std::vector<float> y1 = RunForwardToHost(x1_h, w_h, nullptr, stream, handle);
    const std::vector<float> y2 = RunForwardToHost(x2_h, w_h, nullptr, stream, handle);
    const std::vector<float> y12 = RunForwardToHost(x12_h, w_h, nullptr, stream, handle);

    std::vector<float> expected(y12.size(), 0.0f);
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = y1[i] + y2[i];
    }

    ExpectVectorNear(y12, expected, abs_tol, rel_tol, ReproTag("linearity_no_bias", seed, m, n, k));
}

TEST(LinearForward, InvariantBiasShiftMatchesBiasVector) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    constexpr int m = 19;
    constexpr int n = 29;
    constexpr int k = 13;
    constexpr float abs_tol = 1e-4f;
    constexpr float rel_tol = 1e-4f;

    const uint32_t seed = fa_test::MixSeed(fa_test::kLinearSeedBase, m, n, k, 8);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> x(static_cast<size_t>(m) * k);
    std::vector<float> w(static_cast<size_t>(n) * k);
    std::vector<float> b(static_cast<size_t>(n));
    for (float& v : x) v = dist(rng);
    for (float& v : w) v = dist(rng);
    for (float& v : b) v = dist(rng);

    Stream stream;
    CublasHandle handle;
    Tensor x_h = MakeCpuTensor2D(m, k, x);
    Tensor w_h = MakeCpuTensor2D(n, k, w);
    Tensor b_h = MakeCpuTensor1D(n, b);

    const std::vector<float> y_no_bias = RunForwardToHost(x_h, w_h, nullptr, stream, handle);
    const std::vector<float> y_with_bias = RunForwardToHost(x_h, w_h, &b_h, stream, handle);

    ASSERT_EQ(y_no_bias.size(), y_with_bias.size());
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            const size_t idx = static_cast<size_t>(i) * n + j;
            const float got = y_with_bias[idx] - y_no_bias[idx];
            const float expected = b[static_cast<size_t>(j)];
            const float tol = abs_tol + rel_tol * std::fabs(expected);
            EXPECT_NEAR(got, expected, tol)
                << ReproTag("bias_shift", seed, m, n, k) << " row=" << i << " col=" << j;
        }
    }
}

TEST(LinearForward, InvariantDeterministicForSameInput) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    constexpr int m = 64;
    constexpr int n = 80;
    constexpr int k = 96;
    const uint32_t seed = fa_test::MixSeed(fa_test::kLinearSeedBase, m, n, k, 9);

    fa_test::HostLinearInputs<float> inputs(m, n, k, true, -1.0f, 1.0f, seed);
    ASSERT_TRUE(inputs.b_h.has_value());

    Stream stream;
    CublasHandle handle;

    const std::vector<float> y1 = RunForwardToHost(inputs.x_h, inputs.w_h, &inputs.b_h.value(), stream, handle);
    const std::vector<float> y2 = RunForwardToHost(inputs.x_h, inputs.w_h, &inputs.b_h.value(), stream, handle);

    ASSERT_EQ(y1.size(), y2.size());
    for (size_t i = 0; i < y1.size(); ++i) {
        EXPECT_FLOAT_EQ(y1[i], y2[i]) << ReproTag("determinism", seed, m, n, k) << " idx=" << i;
    }
}

TEST(LinearForward, SingleStreamOrderingReuseStressFixedShape) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    constexpr int launches = 1000;
    constexpr int m = 128;
    constexpr int n = 256;
    constexpr int k = 512;
    constexpr float abs_tol = 1e-5f;
    constexpr float rel_tol = 1e-5f;

    Stream stream;
    CublasHandle handle;
    std::vector<fa_test::QueuedLinearJob> jobs;
    jobs.reserve(launches);

    for (int i = 0; i < launches; ++i) {
        const uint32_t seed = fa_test::MixSeed(fa_test::kLinearSeedBase, m, n, k, i);
        const bool with_bias = fa_test::DeterministicBool(seed);
        fa_test::HostLinearInputs<float> inputs(m, n, k, with_bias, -1.0f, 1.0f, seed);

        Tensor x_d = inputs.x_h.clone(Device::CUDA, stream);
        Tensor w_d = inputs.w_h.clone(Device::CUDA, stream);

        std::optional<Tensor> b_d;
        const Tensor* b_d_ptr = nullptr;
        if (with_bias) {
            ASSERT_TRUE(inputs.b_h.has_value());
            b_d.emplace(inputs.b_h->clone(Device::CUDA, stream));
            b_d_ptr = &b_d.value();
        }

        LinearResults out = linear_forward(x_d, w_d, b_d_ptr, &stream, handle);
        Tensor y_h = out.Y.clone(Device::CPU);

        const std::vector<float>* b_ref = with_bias ? &inputs.b_ref : nullptr;
        const std::vector<float> expected =
            fa_test::reference_linear(inputs.x_ref, inputs.w_ref, b_ref, m, k, n);

        jobs.push_back(fa_test::QueuedLinearJob{
            m,
            n,
            k,
            abs_tol,
            rel_tol,
            std::move(x_d),
            std::move(w_d),
            std::move(b_d),
            std::move(y_h),
            std::move(expected),
        });
    }

    stream.synchronize();
    fa_test::ValidateQueuedJobs(jobs);
}

TEST(LinearForward, SingleStreamOrderingReuseStressShapeCycleABC) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    const fa_test::ShapeCfg cfgs[3] = {
        {31, 47, 29, -1.0f, 1.0f, 1e-4f, 1e-4f, false},
        {96, 80, 64, -1e-6f, 1e-6f, 1e-7f, 1e-4f, true},
        {128, 72, 96, -100.0f, 100.0f, 1.5e-1f, 1e-3f, true},
    };

    constexpr int launches = 600;
    Stream stream;
    CublasHandle handle;
    std::vector<fa_test::QueuedLinearJob> jobs;
    jobs.reserve(launches);

    for (int i = 0; i < launches; ++i) {
        const fa_test::ShapeCfg& c = cfgs[i % 3];
        const uint32_t seed = fa_test::MixSeed(fa_test::kLinearSeedBase, c.m, c.n, c.k, i);
        fa_test::HostLinearInputs<float> inputs(c.m, c.n, c.k, c.with_bias, c.lo, c.hi, seed);

        Tensor x_d = inputs.x_h.clone(Device::CUDA, stream);
        Tensor w_d = inputs.w_h.clone(Device::CUDA, stream);
        std::optional<Tensor> b_d;
        const Tensor* b_d_ptr = nullptr;
        if (c.with_bias) {
            ASSERT_TRUE(inputs.b_h.has_value());
            b_d.emplace(inputs.b_h->clone(Device::CUDA, stream));
            b_d_ptr = &b_d.value();
        }
        LinearResults out = linear_forward(x_d, w_d, b_d_ptr, &stream, handle);
        Tensor y_h = out.Y.clone(Device::CPU);

        const std::vector<float>* b_ref = c.with_bias ? &inputs.b_ref : nullptr;
        const std::vector<float> expected =
            fa_test::reference_linear(inputs.x_ref, inputs.w_ref, b_ref, c.m, c.k, c.n);
        jobs.push_back(fa_test::QueuedLinearJob{
            c.m,
            c.n,
            c.k,
            c.abs_tol,
            c.rel_tol,
            std::move(x_d),
            std::move(w_d),
            std::move(b_d),
            std::move(y_h),
            std::move(expected),
        });
    }

    stream.synchronize();
    fa_test::ValidateQueuedJobs(jobs);
}

}  // namespace
