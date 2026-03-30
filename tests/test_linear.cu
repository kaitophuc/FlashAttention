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
        throw std::invalid_argument("test_linear.cu: Dot inputs must have the same size.");
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

__global__ void affine_inplace_kernel(float* data, int n, float scale, float bias) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * scale + bias;
    }
}

void ApplyAffineInplaceF32(Tensor& t, float scale, float bias, Stream& stream) {
    if (t.dtype_ != DType::F32 || t.device_ != Device::CUDA) {
        throw std::invalid_argument("test_linear.cu: ApplyAffineInplaceF32 expects CUDA float tensor.");
    }
    const int n = static_cast<int>(t.numel());
    const int block = 256;
    const int grid = (n + block - 1) / block;
    affine_inplace_kernel<<<grid, block, 0, stream.s>>>(static_cast<float*>(t.data_), n, scale, bias);
    CUDA_CHECK(cudaGetLastError());
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

TEST(LinearForwardBackward, DemoForwardBackwardCtxStoresXWByPointer) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 30;
    constexpr int n = 40;
    constexpr int k = 20;
    constexpr float abs_tol = 1e-5f;
    constexpr float rel_tol = 1e-5f;
    const uint32_t seed = fa_test::MixSeed(fa_test::kLinearSeedBase, m, n, k, 260);

    fa_test::HostLinearInputs<float> inputs(m, n, k, true, -1.0f, 1.0f, seed);
    ASSERT_TRUE(inputs.b_h.has_value());
    const std::vector<float> dY_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 101u);

    Stream stream;
    CublasHandle handle;

    Tensor x_d = inputs.x_h.clone(Device::CUDA, stream);
    Tensor w_d = inputs.w_h.clone(Device::CUDA, stream);
    Tensor b_d = inputs.b_h->clone(Device::CUDA, stream);

    LinearResults out = linear_forward(x_d, w_d, &b_d, &stream, handle);
    ASSERT_EQ(out.ctx.X, &x_d);
    ASSERT_EQ(out.ctx.W, &w_d);
    ASSERT_TRUE(out.ctx.has_bias);
    ASSERT_EQ(out.ctx.m, m);
    ASSERT_EQ(out.ctx.n, n);
    ASSERT_EQ(out.ctx.k, k);

    Tensor dY_h = MakeCpuTensor2D(m, n, dY_ref);
    Tensor dY_d = dY_h.clone(Device::CUDA, stream);

    LinearGrads grads_before = linear_backward(dY_d, out.ctx, true, true, false, &stream, handle);
    ASSERT_TRUE(grads_before.dX.has_value());
    ASSERT_TRUE(grads_before.dW.has_value());

    std::vector<float> dX_before, dW_before, db_expected_unused;
    fa_test::reference_linear_backward(inputs.x_ref, inputs.w_ref, dY_ref, m, k, n,
                                       dX_before, dW_before, db_expected_unused);
    stream.synchronize();
    ExpectVectorNear(grads_before.dX->clone(Device::CPU).to_vector<float>(), dX_before, abs_tol, rel_tol,
                     ReproTag("ctx_demo_before_dx", seed, m, n, k));
    ExpectVectorNear(grads_before.dW->clone(Device::CPU).to_vector<float>(), dW_before, abs_tol, rel_tol,
                     ReproTag("ctx_demo_before_dw", seed, m, n, k));

    std::vector<float> x2_ref(static_cast<size_t>(m) * k);
    std::vector<float> w2_ref(static_cast<size_t>(n) * k);
    for (size_t i = 0; i < x2_ref.size(); ++i) {
        x2_ref[i] = inputs.x_ref[i] * 0.5f + 0.125f;
    }
    for (size_t i = 0; i < w2_ref.size(); ++i) {
        w2_ref[i] = inputs.w_ref[i] * -0.75f + 0.25f;
    }
    Tensor x2_h = MakeCpuTensor2D(m, k, x2_ref);
    Tensor w2_h = MakeCpuTensor2D(n, k, w2_ref);
    x_d.copy_from(x2_h, stream);
    w_d.copy_from(w2_h, stream);

    LinearGrads grads_after = linear_backward(dY_d, out.ctx, true, true, false, &stream, handle);
    ASSERT_TRUE(grads_after.dX.has_value());
    ASSERT_TRUE(grads_after.dW.has_value());

    std::vector<float> dX_after, dW_after;
    fa_test::reference_linear_backward(x2_ref, w2_ref, dY_ref, m, k, n,
                                       dX_after, dW_after, db_expected_unused);
    stream.synchronize();
    ExpectVectorNear(grads_after.dX->clone(Device::CPU).to_vector<float>(), dX_after, abs_tol, rel_tol,
                     ReproTag("ctx_demo_after_dx", seed, m, n, k));
    ExpectVectorNear(grads_after.dW->clone(Device::CPU).to_vector<float>(), dW_after, abs_tol, rel_tol,
                     ReproTag("ctx_demo_after_dw", seed, m, n, k));
}

TEST(LinearForwardBackward, TwoStageReuseNoMidTransfer) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 7;
    constexpr int n = 9;
    constexpr int k = 5;
    constexpr float abs_tol = 1e-4f;
    constexpr float rel_tol = 1e-4f;
    const uint32_t seed = fa_test::MixSeed(fa_test::kLinearSeedBase, m, n, k, 320);

    fa_test::HostLinearInputs<float> inputs(m, n, k, true, -1.0f, 1.0f, seed);
    ASSERT_TRUE(inputs.b_h.has_value());
    const std::vector<float> dY1_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 1u);
    const std::vector<float> dY2_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 2u);

    Stream stream;
    CublasHandle handle;

    // H2D setup only.
    Tensor x_d = inputs.x_h.clone(Device::CUDA, stream);
    Tensor w_d = inputs.w_h.clone(Device::CUDA, stream);
    Tensor b_d = inputs.b_h->clone(Device::CUDA, stream);
    Tensor dY1_d = MakeCpuTensor2D(m, n, dY1_ref).clone(Device::CUDA, stream);
    Tensor dY2_d = MakeCpuTensor2D(m, n, dY2_ref).clone(Device::CUDA, stream);

    // Stage 1 entirely on device.
    LinearResults out1 = linear_forward(x_d, w_d, &b_d, &stream, handle);
    LinearGrads g1 = linear_backward(dY1_d, out1.ctx, true, true, true, &stream, handle);
    ASSERT_TRUE(g1.dX.has_value());
    ASSERT_TRUE(g1.dW.has_value());
    ASSERT_TRUE(g1.db.has_value());

    // Device-only mutation.
    ApplyAffineInplaceF32(x_d, 0.5f, 0.125f, stream);
    ApplyAffineInplaceF32(w_d, -0.75f, 0.25f, stream);
    ApplyAffineInplaceF32(b_d, 0.9f, -0.05f, stream);

    // Stage 2 entirely on device.
    LinearResults out2 = linear_forward(x_d, w_d, &b_d, &stream, handle);
    LinearGrads g2 = linear_backward(dY2_d, out2.ctx, true, true, true, &stream, handle);
    ASSERT_TRUE(g2.dX.has_value());
    ASSERT_TRUE(g2.dW.has_value());
    ASSERT_TRUE(g2.db.has_value());

    // Transfer only at the end.
    stream.synchronize();
    const std::vector<float> g1_dX = g1.dX->clone(Device::CPU).to_vector<float>();
    const std::vector<float> g1_dW = g1.dW->clone(Device::CPU).to_vector<float>();
    const std::vector<float> g1_db = g1.db->clone(Device::CPU).to_vector<float>();
    const std::vector<float> g2_dX = g2.dX->clone(Device::CPU).to_vector<float>();
    const std::vector<float> g2_dW = g2.dW->clone(Device::CPU).to_vector<float>();
    const std::vector<float> g2_db = g2.db->clone(Device::CPU).to_vector<float>();

    std::vector<float> g1_dX_expected, g1_dW_expected, g1_db_expected;
    fa_test::reference_linear_backward(inputs.x_ref, inputs.w_ref, dY1_ref, m, k, n,
                                       g1_dX_expected, g1_dW_expected, g1_db_expected);

    std::vector<float> x2_ref = inputs.x_ref;
    std::vector<float> w2_ref = inputs.w_ref;
    std::vector<float> b2_ref = inputs.b_ref;
    for (float& v : x2_ref) v = v * 0.5f + 0.125f;
    for (float& v : w2_ref) v = v * -0.75f + 0.25f;
    for (float& v : b2_ref) v = v * 0.9f - 0.05f;

    std::vector<float> g2_dX_expected, g2_dW_expected, g2_db_expected;
    fa_test::reference_linear_backward(x2_ref, w2_ref, dY2_ref, m, k, n,
                                       g2_dX_expected, g2_dW_expected, g2_db_expected);

    ExpectVectorNear(g1_dX, g1_dX_expected, abs_tol, rel_tol, ReproTag("combined_2stage_s1_dx", seed, m, n, k));
    ExpectVectorNear(g1_dW, g1_dW_expected, abs_tol, rel_tol, ReproTag("combined_2stage_s1_dw", seed, m, n, k));
    ExpectVectorNear(g1_db, g1_db_expected, abs_tol, rel_tol, ReproTag("combined_2stage_s1_db", seed, m, n, k));
    ExpectVectorNear(g2_dX, g2_dX_expected, abs_tol, rel_tol, ReproTag("combined_2stage_s2_dx", seed, m, n, k));
    ExpectVectorNear(g2_dW, g2_dW_expected, abs_tol, rel_tol, ReproTag("combined_2stage_s2_dw", seed, m, n, k));
    ExpectVectorNear(g2_db, g2_db_expected, abs_tol, rel_tol, ReproTag("combined_2stage_s2_db", seed, m, n, k));
}

TEST(LinearForwardBackward, CtxIsolationAcrossMultipleForwardsNoMidTransfer) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m1 = 5, n1 = 7, k1 = 3;
    constexpr int m2 = 6, n2 = 4, k2 = 5;
    constexpr float abs_tol = 1e-4f;
    constexpr float rel_tol = 1e-4f;
    const uint32_t seed1 = fa_test::MixSeed(fa_test::kLinearSeedBase, m1, n1, k1, 321);
    const uint32_t seed2 = fa_test::MixSeed(fa_test::kLinearSeedBase, m2, n2, k2, 322);

    fa_test::HostLinearInputs<float> a(m1, n1, k1, true, -1.0f, 1.0f, seed1);
    fa_test::HostLinearInputs<float> b(m2, n2, k2, false, -1.0f, 1.0f, seed2);
    ASSERT_TRUE(a.b_h.has_value());
    const std::vector<float> dYa_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m1) * n1, -1.0f, 1.0f, seed1 + 1u);
    const std::vector<float> dYb_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m2) * n2, -1.0f, 1.0f, seed2 + 1u);

    Stream stream;
    CublasHandle handle;

    // H2D setup only.
    Tensor xa_d = a.x_h.clone(Device::CUDA, stream);
    Tensor wa_d = a.w_h.clone(Device::CUDA, stream);
    Tensor ba_d = a.b_h->clone(Device::CUDA, stream);
    Tensor xb_d = b.x_h.clone(Device::CUDA, stream);
    Tensor wb_d = b.w_h.clone(Device::CUDA, stream);
    Tensor dYa_d = MakeCpuTensor2D(m1, n1, dYa_ref).clone(Device::CUDA, stream);
    Tensor dYb_d = MakeCpuTensor2D(m2, n2, dYb_ref).clone(Device::CUDA, stream);

    LinearResults out_a = linear_forward(xa_d, wa_d, &ba_d, &stream, handle);
    LinearResults out_b = linear_forward(xb_d, wb_d, nullptr, &stream, handle);

    // Backward in reverse order to stress ctx isolation.
    LinearGrads gb = linear_backward(dYb_d, out_b.ctx, true, true, true, &stream, handle);
    LinearGrads ga = linear_backward(dYa_d, out_a.ctx, true, true, true, &stream, handle);
    ASSERT_TRUE(gb.dX.has_value());
    ASSERT_TRUE(gb.dW.has_value());
    ASSERT_FALSE(gb.db.has_value());
    ASSERT_TRUE(ga.dX.has_value());
    ASSERT_TRUE(ga.dW.has_value());
    ASSERT_TRUE(ga.db.has_value());

    // Transfer only at the end.
    stream.synchronize();
    const std::vector<float> gb_dX = gb.dX->clone(Device::CPU).to_vector<float>();
    const std::vector<float> gb_dW = gb.dW->clone(Device::CPU).to_vector<float>();
    const std::vector<float> ga_dX = ga.dX->clone(Device::CPU).to_vector<float>();
    const std::vector<float> ga_dW = ga.dW->clone(Device::CPU).to_vector<float>();
    const std::vector<float> ga_db = ga.db->clone(Device::CPU).to_vector<float>();

    std::vector<float> gb_dX_expected, gb_dW_expected, gb_db_expected_unused;
    std::vector<float> ga_dX_expected, ga_dW_expected, ga_db_expected;
    fa_test::reference_linear_backward(b.x_ref, b.w_ref, dYb_ref, m2, k2, n2,
                                       gb_dX_expected, gb_dW_expected, gb_db_expected_unused);
    fa_test::reference_linear_backward(a.x_ref, a.w_ref, dYa_ref, m1, k1, n1,
                                       ga_dX_expected, ga_dW_expected, ga_db_expected);

    ExpectVectorNear(gb_dX, gb_dX_expected, abs_tol, rel_tol, ReproTag("combined_iso_b_dx", seed2, m2, n2, k2));
    ExpectVectorNear(gb_dW, gb_dW_expected, abs_tol, rel_tol, ReproTag("combined_iso_b_dw", seed2, m2, n2, k2));
    ExpectVectorNear(ga_dX, ga_dX_expected, abs_tol, rel_tol, ReproTag("combined_iso_a_dx", seed1, m1, n1, k1));
    ExpectVectorNear(ga_dW, ga_dW_expected, abs_tol, rel_tol, ReproTag("combined_iso_a_dw", seed1, m1, n1, k1));
    ExpectVectorNear(ga_db, ga_db_expected, abs_tol, rel_tol, ReproTag("combined_iso_a_db", seed1, m1, n1, k1));
}

TEST(LinearForwardBackward, SweepAllCasesNoMidTransfer) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    const std::vector<fa_test::LinearForwardBackwardCase> cases = fa_test::BuildForwardBackwardCases();
    std::vector<std::string> failures;
    failures.reserve(64);

    for (const fa_test::LinearForwardBackwardCase& c : cases) {
        for (int iter = 0; iter < c.iters; ++iter) {
            const uint32_t seed = fa_test::MixSeed(fa_test::kLinearSeedBase, c.m, c.n, c.k, 700 + iter);
            fa_test::HostLinearInputs<float> inputs(c.m, c.n, c.k, c.with_bias, c.lo, c.hi, seed);
            const std::vector<float> dY1_ref = fa_test::SampleUniformVector(
                static_cast<size_t>(c.m) * c.n, c.lo, c.hi, seed + 1u);
            const std::vector<float> dY2_ref = fa_test::SampleUniformVector(
                static_cast<size_t>(c.m) * c.n, c.lo, c.hi, seed + 2u);

            Stream stream;
            CublasHandle handle;

            // H2D setup only.
            Tensor x_d = inputs.x_h.clone(Device::CUDA, stream);
            Tensor w_d = inputs.w_h.clone(Device::CUDA, stream);
            std::optional<Tensor> b_d;
            const Tensor* b_d_ptr = nullptr;
            if (c.with_bias) {
                ASSERT_TRUE(inputs.b_h.has_value());
                b_d.emplace(inputs.b_h->clone(Device::CUDA, stream));
                b_d_ptr = &b_d.value();
            }
            Tensor dY1_d = MakeCpuTensor2D(c.m, c.n, dY1_ref).clone(Device::CUDA, stream);
            Tensor dY2_d = MakeCpuTensor2D(c.m, c.n, dY2_ref).clone(Device::CUDA, stream);

            // Stage 1 entirely on device.
            LinearResults out1 = linear_forward(x_d, w_d, b_d_ptr, &stream, handle);
            LinearGrads g1 = linear_backward(dY1_d, out1.ctx, true, true, c.with_bias, &stream, handle);
            ASSERT_TRUE(g1.dX.has_value());
            ASSERT_TRUE(g1.dW.has_value());
            if (c.with_bias) {
                ASSERT_TRUE(g1.db.has_value());
            } else {
                ASSERT_FALSE(g1.db.has_value());
            }

            // Device-only mutation.
            ApplyAffineInplaceF32(x_d, 0.5f, 0.125f, stream);
            ApplyAffineInplaceF32(w_d, -0.75f, 0.25f, stream);
            if (c.with_bias) {
                ApplyAffineInplaceF32(b_d.value(), 0.9f, -0.05f, stream);
            }

            // Stage 2 entirely on device.
            LinearResults out2 = linear_forward(x_d, w_d, b_d_ptr, &stream, handle);
            LinearGrads g2 = linear_backward(dY2_d, out2.ctx, true, true, c.with_bias, &stream, handle);
            ASSERT_TRUE(g2.dX.has_value());
            ASSERT_TRUE(g2.dW.has_value());
            if (c.with_bias) {
                ASSERT_TRUE(g2.db.has_value());
            } else {
                ASSERT_FALSE(g2.db.has_value());
            }

            // Transfer only at the end.
            stream.synchronize();
            const std::vector<float> g1_dX = g1.dX->clone(Device::CPU).to_vector<float>();
            const std::vector<float> g1_dW = g1.dW->clone(Device::CPU).to_vector<float>();
            const std::vector<float> g2_dX = g2.dX->clone(Device::CPU).to_vector<float>();
            const std::vector<float> g2_dW = g2.dW->clone(Device::CPU).to_vector<float>();

            std::vector<float> g1_db;
            std::vector<float> g2_db;
            if (c.with_bias) {
                g1_db = g1.db->clone(Device::CPU).to_vector<float>();
                g2_db = g2.db->clone(Device::CPU).to_vector<float>();
            }

            std::vector<float> g1_dX_expected, g1_dW_expected, g1_db_expected;
            fa_test::reference_linear_backward(inputs.x_ref, inputs.w_ref, dY1_ref, c.m, c.k, c.n,
                                               g1_dX_expected, g1_dW_expected, g1_db_expected);

            std::vector<float> x2_ref = inputs.x_ref;
            std::vector<float> w2_ref = inputs.w_ref;
            for (float& v : x2_ref) v = v * 0.5f + 0.125f;
            for (float& v : w2_ref) v = v * -0.75f + 0.25f;

            std::vector<float> b2_ref;
            if (c.with_bias) {
                b2_ref = inputs.b_ref;
                for (float& v : b2_ref) v = v * 0.9f - 0.05f;
            }

            std::vector<float> g2_dX_expected, g2_dW_expected, g2_db_expected;
            fa_test::reference_linear_backward(x2_ref, w2_ref, dY2_ref, c.m, c.k, c.n,
                                               g2_dX_expected, g2_dW_expected, g2_db_expected);

            auto first_fail_idx = [&](const std::vector<float>& got, const std::vector<float>& expected) -> int {
                if (got.size() != expected.size()) return -2;
                for (size_t i = 0; i < got.size(); ++i) {
                    const float tol = c.abs_tol + c.rel_tol * std::fabs(expected[i]);
                    if (std::fabs(got[i] - expected[i]) > tol) {
                        return static_cast<int>(i);
                    }
                }
                return -1;
            };

            const int s1_dx_fail = first_fail_idx(g1_dX, g1_dX_expected);
            const int s1_dw_fail = first_fail_idx(g1_dW, g1_dW_expected);
            const int s2_dx_fail = first_fail_idx(g2_dX, g2_dX_expected);
            const int s2_dw_fail = first_fail_idx(g2_dW, g2_dW_expected);
            int s1_db_fail = -1;
            int s2_db_fail = -1;
            if (c.with_bias) {
                s1_db_fail = first_fail_idx(g1_db, g1_db_expected);
                s2_db_fail = first_fail_idx(g2_db, g2_db_expected);
            }

            const bool failed = (s1_dx_fail != -1) || (s1_dw_fail != -1) ||
                                (s2_dx_fail != -1) || (s2_dw_fail != -1) ||
                                (c.with_bias && ((s1_db_fail != -1) || (s2_db_fail != -1)));
            if (failed) {
                std::ostringstream one;
                one << "case=" << c.name
                    << " dist=" << fa_test::DistName(c.dist)
                    << " bias=" << (c.with_bias ? "true" : "false")
                    << " iter=" << iter
                    << " seed=" << seed
                    << " m=" << c.m << " n=" << c.n << " k=" << c.k
                    << " s1_dx_fail_idx=" << s1_dx_fail
                    << " s1_dw_fail_idx=" << s1_dw_fail
                    << " s2_dx_fail_idx=" << s2_dx_fail
                    << " s2_dw_fail_idx=" << s2_dw_fail;
                if (c.with_bias) {
                    one << " s1_db_fail_idx=" << s1_db_fail
                        << " s2_db_fail_idx=" << s2_db_fail;
                }
                failures.push_back(one.str());
            }
        }
    }

    if (!failures.empty()) {
        std::ostringstream all;
        all << "LinearForwardBackward sweep failed in " << failures.size() << " case(s):\n";
        for (const std::string& f : failures) {
            all << "  " << f << "\n";
        }
        ADD_FAILURE() << all.str();
    }
}

TEST(LinearForwardBackward, MegaCoverageMatrixNoMidTransfer) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 8;
    constexpr int n = 6;
    constexpr int k = 5;
    constexpr float abs_tol = 1e-4f;
    constexpr float rel_tol = 1e-4f;
    constexpr float zero_abs_tol = 1e-7f;
    const uint32_t seed = fa_test::MixSeed(fa_test::kLinearSeedBase, m, n, k, 390);

    fa_test::HostLinearInputs<float> inputs(m, n, k, true, -1.0f, 1.0f, seed);
    ASSERT_TRUE(inputs.b_h.has_value());
    const std::vector<float> dY_rand_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 1u);
    const std::vector<float> dY_zero_ref(static_cast<size_t>(m) * n, 0.0f);

    Stream stream;
    CublasHandle handle;

    // H2D setup only.
    Tensor x_d = inputs.x_h.clone(Device::CUDA, stream);
    Tensor w_d = inputs.w_h.clone(Device::CUDA, stream);
    Tensor b_d = inputs.b_h->clone(Device::CUDA, stream);
    Tensor dY_rand_d = MakeCpuTensor2D(m, n, dY_rand_ref).clone(Device::CUDA, stream);
    Tensor dY_zero_d = MakeCpuTensor2D(m, n, dY_zero_ref).clone(Device::CUDA, stream);

    // Forward with and without bias from the same X/W.
    LinearResults out_bias = linear_forward(x_d, w_d, &b_d, &stream, handle);
    LinearResults out_no_bias = linear_forward(x_d, w_d, nullptr, &stream, handle);

    // Backward matrix over flag combinations.
    LinearGrads g_all = linear_backward(dY_rand_d, out_bias.ctx, true, true, true, &stream, handle);
    LinearGrads g_all_rep = linear_backward(dY_rand_d, out_bias.ctx, true, true, true, &stream, handle);
    LinearGrads g_dx_only = linear_backward(dY_rand_d, out_bias.ctx, true, false, false, &stream, handle);
    LinearGrads g_dw_only = linear_backward(dY_rand_d, out_bias.ctx, false, true, false, &stream, handle);
    LinearGrads g_db_only = linear_backward(dY_rand_d, out_bias.ctx, false, false, true, &stream, handle);
    LinearGrads g_none = linear_backward(dY_rand_d, out_bias.ctx, false, false, false, &stream, handle);
    LinearGrads g_no_bias_needs_db = linear_backward(dY_rand_d, out_no_bias.ctx, false, false, true, &stream, handle);
    LinearGrads g_no_bias_all = linear_backward(dY_rand_d, out_no_bias.ctx, true, true, true, &stream, handle);
    LinearGrads g_zero = linear_backward(dY_zero_d, out_bias.ctx, true, true, true, &stream, handle);

    // Flag/presence checks.
    ASSERT_TRUE(g_all.dX.has_value());
    ASSERT_TRUE(g_all.dW.has_value());
    ASSERT_TRUE(g_all.db.has_value());
    ASSERT_TRUE(g_all_rep.dX.has_value());
    ASSERT_TRUE(g_all_rep.dW.has_value());
    ASSERT_TRUE(g_all_rep.db.has_value());

    ASSERT_TRUE(g_dx_only.has_dX);
    ASSERT_FALSE(g_dx_only.has_dW);
    ASSERT_FALSE(g_dx_only.has_db);
    ASSERT_TRUE(g_dx_only.dX.has_value());
    ASSERT_FALSE(g_dx_only.dW.has_value());
    ASSERT_FALSE(g_dx_only.db.has_value());

    ASSERT_FALSE(g_dw_only.has_dX);
    ASSERT_TRUE(g_dw_only.has_dW);
    ASSERT_FALSE(g_dw_only.has_db);
    ASSERT_FALSE(g_dw_only.dX.has_value());
    ASSERT_TRUE(g_dw_only.dW.has_value());
    ASSERT_FALSE(g_dw_only.db.has_value());

    ASSERT_FALSE(g_db_only.has_dX);
    ASSERT_FALSE(g_db_only.has_dW);
    ASSERT_TRUE(g_db_only.has_db);
    ASSERT_FALSE(g_db_only.dX.has_value());
    ASSERT_FALSE(g_db_only.dW.has_value());
    ASSERT_TRUE(g_db_only.db.has_value());

    ASSERT_FALSE(g_none.has_dX);
    ASSERT_FALSE(g_none.has_dW);
    ASSERT_FALSE(g_none.has_db);
    ASSERT_FALSE(g_none.dX.has_value());
    ASSERT_FALSE(g_none.dW.has_value());
    ASSERT_FALSE(g_none.db.has_value());

    ASSERT_FALSE(g_no_bias_needs_db.has_db);
    ASSERT_FALSE(g_no_bias_needs_db.db.has_value());
    ASSERT_TRUE(g_no_bias_all.has_dX);
    ASSERT_TRUE(g_no_bias_all.has_dW);
    ASSERT_FALSE(g_no_bias_all.has_db);
    ASSERT_TRUE(g_no_bias_all.dX.has_value());
    ASSERT_TRUE(g_no_bias_all.dW.has_value());
    ASSERT_FALSE(g_no_bias_all.db.has_value());

    ASSERT_TRUE(g_zero.dX.has_value());
    ASSERT_TRUE(g_zero.dW.has_value());
    ASSERT_TRUE(g_zero.db.has_value());

    // Reference expectations on host.
    std::vector<float> dX_expected, dW_expected, db_expected;
    fa_test::reference_linear_backward(inputs.x_ref, inputs.w_ref, dY_rand_ref, m, k, n,
                                       dX_expected, dW_expected, db_expected);
    std::vector<float> dX_zero_expected(static_cast<size_t>(m) * k, 0.0f);
    std::vector<float> dW_zero_expected(static_cast<size_t>(n) * k, 0.0f);
    std::vector<float> db_zero_expected(static_cast<size_t>(n), 0.0f);

    // Transfer only at the end.
    stream.synchronize();

    const std::vector<float> all_dX = g_all.dX->clone(Device::CPU).to_vector<float>();
    const std::vector<float> all_dW = g_all.dW->clone(Device::CPU).to_vector<float>();
    const std::vector<float> all_db = g_all.db->clone(Device::CPU).to_vector<float>();
    const std::vector<float> rep_dX = g_all_rep.dX->clone(Device::CPU).to_vector<float>();
    const std::vector<float> rep_dW = g_all_rep.dW->clone(Device::CPU).to_vector<float>();
    const std::vector<float> rep_db = g_all_rep.db->clone(Device::CPU).to_vector<float>();
    const std::vector<float> dx_only = g_dx_only.dX->clone(Device::CPU).to_vector<float>();
    const std::vector<float> dw_only = g_dw_only.dW->clone(Device::CPU).to_vector<float>();
    const std::vector<float> db_only = g_db_only.db->clone(Device::CPU).to_vector<float>();
    const std::vector<float> nb_dX = g_no_bias_all.dX->clone(Device::CPU).to_vector<float>();
    const std::vector<float> nb_dW = g_no_bias_all.dW->clone(Device::CPU).to_vector<float>();
    const std::vector<float> z_dX = g_zero.dX->clone(Device::CPU).to_vector<float>();
    const std::vector<float> z_dW = g_zero.dW->clone(Device::CPU).to_vector<float>();
    const std::vector<float> z_db = g_zero.db->clone(Device::CPU).to_vector<float>();

    // Value checks for requested grads.
    ExpectVectorNear(all_dX, dX_expected, abs_tol, rel_tol, ReproTag("mega_all_dx", seed, m, n, k));
    ExpectVectorNear(all_dW, dW_expected, abs_tol, rel_tol, ReproTag("mega_all_dw", seed, m, n, k));
    ExpectVectorNear(all_db, db_expected, abs_tol, rel_tol, ReproTag("mega_all_db", seed, m, n, k));
    ExpectVectorNear(dx_only, dX_expected, abs_tol, rel_tol, ReproTag("mega_dx_only", seed, m, n, k));
    ExpectVectorNear(dw_only, dW_expected, abs_tol, rel_tol, ReproTag("mega_dw_only", seed, m, n, k));
    ExpectVectorNear(db_only, db_expected, abs_tol, rel_tol, ReproTag("mega_db_only", seed, m, n, k));

    // Determinism for repeated backward from same ctx and dY.
    ASSERT_EQ(all_dX.size(), rep_dX.size());
    ASSERT_EQ(all_dW.size(), rep_dW.size());
    ASSERT_EQ(all_db.size(), rep_db.size());
    for (size_t i = 0; i < all_dX.size(); ++i) {
        EXPECT_FLOAT_EQ(all_dX[i], rep_dX[i]) << ReproTag("mega_det_dx", seed, m, n, k) << " idx=" << i;
    }
    for (size_t i = 0; i < all_dW.size(); ++i) {
        EXPECT_FLOAT_EQ(all_dW[i], rep_dW[i]) << ReproTag("mega_det_dw", seed, m, n, k) << " idx=" << i;
    }
    for (size_t i = 0; i < all_db.size(); ++i) {
        EXPECT_FLOAT_EQ(all_db[i], rep_db[i]) << ReproTag("mega_det_db", seed, m, n, k) << " idx=" << i;
    }

    // Bias-independence: dX and dW should match with/without bias.
    ExpectVectorNear(nb_dX, all_dX, abs_tol, rel_tol, ReproTag("mega_bias_ind_dx", seed, m, n, k));
    ExpectVectorNear(nb_dW, all_dW, abs_tol, rel_tol, ReproTag("mega_bias_ind_dw", seed, m, n, k));

    // Zero dY should produce zeros.
    ExpectVectorNear(z_dX, dX_zero_expected, zero_abs_tol, 0.0f, ReproTag("mega_zero_dx", seed, m, n, k));
    ExpectVectorNear(z_dW, dW_zero_expected, zero_abs_tol, 0.0f, ReproTag("mega_zero_dw", seed, m, n, k));
    ExpectVectorNear(z_db, db_zero_expected, zero_abs_tol, 0.0f, ReproTag("mega_zero_db", seed, m, n, k));
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

    EXPECT_NO_THROW((void)linear_backward(dY, out.ctx, true, true, false, &non_default_stream, handle));

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

    const int launches = fa_test::LongStressEnabled() ? 1400 : 500;
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

    const int launches = fa_test::LongStressEnabled() ? 900 : 360;
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
    EXPECT_NO_THROW((void)linear_forward(x, w, nullptr, &non_default_stream, handle));

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

    constexpr int launches = 1400;
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

    constexpr int launches = 900;
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
