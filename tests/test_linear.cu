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

    struct EdgeCase {
        const char* name;
        int m;
        int n;
        int k;
        bool with_bias;
        float abs_tol;
        float rel_tol;
        std::vector<float> x;
        std::vector<float> w;
        std::vector<float> b;
    };

    std::vector<EdgeCase> cases;

    // Degenerate all-zero case (m=n=k=1).
    cases.push_back(EdgeCase{"zeros_111", 1, 1, 1, true, 1e-7f, 1e-7f, {0.0f}, {0.0f}, {0.0f}});

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
        cases.push_back(EdgeCase{"one_hot_identity", m, n, k, true, 1e-6f, 1e-6f, x, w, b});
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
        cases.push_back(EdgeCase{"mixed_mag_cancellation", m, n, k, true, 2.0e-2f, 2.0e-4f, x, w, b});
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
        cases.push_back(EdgeCase{"tall_skinny", m, n, k, true, 1e-4f, 1e-4f, x, w, b});
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
        cases.push_back(EdgeCase{"short_fat", m, n, k, true, 1e-4f, 1e-4f, x, w, b});
    }

    for (const EdgeCase& c : cases) {
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
