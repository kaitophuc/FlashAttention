#include "general.h"
#include "test_relu.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace {

std::string ReproTag(const std::string& name, uint32_t seed, int m, int n) {
    std::ostringstream os;
    os << "case=" << name << " seed=" << seed << " m=" << m << " n=" << n;
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

Tensor MakeCpuTensor2D(int rows, int cols, const std::vector<float>& values) {
    Tensor t({rows, cols}, DType::F32, Device::CPU);
    t.copy_from(values);
    return t;
}

std::vector<float> RunForwardToHost(const Tensor& x_h, Stream& stream) {
    Tensor x_d = x_h.clone(Device::CUDA, stream);
    ReluResults out = relu_forward(x_d, &stream);
    Tensor y_h = out.Y.clone(Device::CPU);
    stream.synchronize();
    return y_h.to_vector<float>();
}

float Dot(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("test_relu.cu: Dot inputs must have the same size.");
    }
    double acc = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        acc += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return static_cast<float>(acc);
}

float ForwardLossDotDY(const std::vector<float>& x,
                       const std::vector<float>& dY_ref,
                       int m,
                       int n,
                       Stream& stream) {
    Tensor x_h = MakeCpuTensor2D(m, n, x);
    const std::vector<float> y = RunForwardToHost(x_h, stream);
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
        throw std::invalid_argument("test_relu.cu: ApplyAffineInplaceF32 expects CUDA float tensor.");
    }
    const int n = static_cast<int>(t.numel());
    const int block = 256;
    const int grid = (n + block - 1) / block;
    affine_inplace_kernel<<<grid, block, 0, stream.s>>>(static_cast<float*>(t.data_), n, scale, bias);
    CUDA_CHECK(cudaGetLastError());
}

TEST(ReluForward, RejectsNullStream) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }
    Stream stream;
    Tensor x({4, 7}, DType::F32, Device::CUDA, stream);
    EXPECT_THROW((void)relu_forward(x, nullptr), std::invalid_argument);
}

TEST(ReluForward, RejectsNonDefaultStream) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }
    Stream stream;
    Tensor x({4, 7}, DType::F32, Device::CUDA, stream);

    cudaStream_t raw_non_default = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&raw_non_default, cudaStreamNonBlocking));
    Stream non_default_stream;
    non_default_stream.s = raw_non_default;
    non_default_stream.owns_ = false;

    EXPECT_NO_THROW((void)relu_forward(x, &non_default_stream));

    CUDA_CHECK(cudaStreamDestroy(raw_non_default));
}

TEST(ReluForward, RejectsNonF32) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }
    Stream stream;
    Tensor x({4, 7}, DType::F16, Device::CUDA, stream);
    EXPECT_THROW((void)relu_forward(x, &stream), std::invalid_argument);
}

TEST(ReluForward, RejectsNon2DInput) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }
    Stream stream;
    Tensor x({4, 7, 2}, DType::F32, Device::CUDA, stream);
    EXPECT_THROW((void)relu_forward(x, &stream), std::invalid_argument);
}

TEST(ReluForward, RejectsNonCudaInput) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }
    Stream stream;
    Tensor x({4, 7}, DType::F32, Device::CPU);
    EXPECT_THROW((void)relu_forward(x, &stream), std::invalid_argument);
}

TEST(ReluBackward, RejectsNullStream) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }
    Stream stream;
    Tensor x({4, 7}, DType::F32, Device::CUDA, stream);
    ReluResults out = relu_forward(x, &stream);
    Tensor dY({4, 7}, DType::F32, Device::CUDA, stream);

    EXPECT_THROW((void)relu_backward(dY, out.ctx, nullptr), std::invalid_argument);
}

TEST(ReluBackward, RejectsNonDefaultStream) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }
    Stream stream;
    Tensor x({4, 7}, DType::F32, Device::CUDA, stream);
    ReluResults out = relu_forward(x, &stream);
    Tensor dY({4, 7}, DType::F32, Device::CUDA, stream);

    cudaStream_t raw_non_default = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&raw_non_default, cudaStreamNonBlocking));
    Stream non_default_stream;
    non_default_stream.s = raw_non_default;
    non_default_stream.owns_ = false;

    EXPECT_NO_THROW((void)relu_backward(dY, out.ctx, &non_default_stream));

    CUDA_CHECK(cudaStreamDestroy(raw_non_default));
}

TEST(ReluBackward, RejectsNonF32DY) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }
    Stream stream;
    Tensor x({4, 7}, DType::F32, Device::CUDA, stream);
    ReluResults out = relu_forward(x, &stream);
    Tensor dY({4, 7}, DType::F16, Device::CUDA, stream);

    EXPECT_THROW((void)relu_backward(dY, out.ctx, &stream), std::invalid_argument);
}

TEST(ReluBackward, RejectsDYShapeMismatch) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }
    Stream stream;
    Tensor x({4, 7}, DType::F32, Device::CUDA, stream);
    ReluResults out = relu_forward(x, &stream);
    Tensor dY({7, 4}, DType::F32, Device::CUDA, stream);

    EXPECT_THROW((void)relu_backward(dY, out.ctx, &stream), std::invalid_argument);
}

TEST(ReluBackward, RejectsNullCtxPointer) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }
    Stream stream;
    Tensor dY({4, 7}, DType::F32, Device::CUDA, stream);
    const ReluCtx bad_ctx{nullptr};

    EXPECT_THROW((void)relu_backward(dY, bad_ctx, &stream), std::invalid_argument);
}

TEST(ReluBackward, RejectsDTypeMismatchCtx) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }
    Stream stream;
    Tensor x({4, 7}, DType::F16, Device::CUDA, stream);
    const ReluCtx ctx{&x};
    Tensor dY({4, 7}, DType::F32, Device::CUDA, stream);

    EXPECT_THROW((void)relu_backward(dY, ctx, &stream), std::invalid_argument);
}

TEST(ReluBackward, RejectsDeviceMismatchCtx) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }
    Stream stream;
    Tensor x({4, 7}, DType::F32, Device::CPU);
    const ReluCtx ctx{&x};
    Tensor dY({4, 7}, DType::F32, Device::CUDA, stream);

    EXPECT_THROW((void)relu_backward(dY, ctx, &stream), std::invalid_argument);
}

TEST(ReluBackward, RejectsNon2DDY) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }
    Stream stream;
    Tensor x({4, 7}, DType::F32, Device::CUDA, stream);
    const ReluCtx ctx{&x};
    Tensor dY({4, 7, 2}, DType::F32, Device::CUDA, stream);

    EXPECT_THROW((void)relu_backward(dY, ctx, &stream), std::invalid_argument);
}

TEST(ReluBackward, RejectsNon2DCtxX) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }
    Stream stream;
    Tensor x({4, 7, 2}, DType::F32, Device::CUDA, stream);
    const ReluCtx ctx{&x};
    Tensor dY({56, 1}, DType::F32, Device::CUDA, stream);

    EXPECT_THROW((void)relu_backward(dY, ctx, &stream), std::invalid_argument);
}

TEST(ReluBackward, MatchesReferenceOddShape) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 13;
    constexpr int n = 37;
    constexpr float abs_tol = 1e-6f;
    constexpr float rel_tol = 1e-6f;

    const uint32_t seed = fa_test::MixSeed(fa_test::kReluSeedBase, m, n, 0, 101);
    const std::vector<float> x_ref = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed);
    const std::vector<float> dY_ref = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 1u);

    Stream stream;
    Tensor x_h = MakeCpuTensor2D(m, n, x_ref);
    Tensor dY_h = MakeCpuTensor2D(m, n, dY_ref);
    Tensor x_d = x_h.clone(Device::CUDA, stream);
    Tensor dY_d = dY_h.clone(Device::CUDA, stream);

    ReluResults out = relu_forward(x_d, &stream);
    ReluGrads grads = relu_backward(dY_d, out.ctx, &stream);

    const std::vector<float> expected = fa_test::reference_relu_backward(dY_ref, x_ref);
    stream.synchronize();
    ExpectVectorNear(grads.dX.clone(Device::CPU).to_vector<float>(), expected, abs_tol, rel_tol,
                     ReproTag("backward_odd", seed, m, n));
}

TEST(ReluForward, SweepAllCases) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    const std::vector<fa_test::ReluCase> cases = fa_test::BuildForwardCases();
    std::vector<std::string> failures;
    failures.reserve(64);

    for (const auto& c : cases) {
        for (int iter = 0; iter < c.iters; ++iter) {
            const uint32_t seed = fa_test::MixSeed(fa_test::kReluSeedBase, c.m, c.n, iter, 0);
            const std::vector<float> x_ref =
                fa_test::SampleUniformVector(static_cast<size_t>(c.m) * c.n, c.lo, c.hi, seed);

            Stream stream;
            Tensor x_h = MakeCpuTensor2D(c.m, c.n, x_ref);
            Tensor x_d = x_h.clone(Device::CUDA, stream);

            ReluResults out = relu_forward(x_d, &stream);
            Tensor y_h = out.Y.clone(Device::CPU);
            stream.synchronize();

            const std::vector<float> got = y_h.to_vector<float>();
            const std::vector<float> expected = fa_test::reference_relu_forward(x_ref);

            int fail_count = 0;
            float worst_abs_err = 0.0f;
            float worst_tol = 0.0f;
            int worst_idx = -1;
            for (size_t i = 0; i < got.size(); ++i) {
                const float abs_err = std::fabs(got[i] - expected[i]);
                const float tol = c.abs_tol + c.rel_tol * std::fabs(expected[i]);
                if (abs_err > tol) {
                    ++fail_count;
                    if (abs_err > worst_abs_err) {
                        worst_abs_err = abs_err;
                        worst_tol = tol;
                        worst_idx = static_cast<int>(i);
                    }
                }
            }

            if (fail_count > 0) {
                std::ostringstream one;
                one << "case=" << c.name
                    << " dist=" << fa_test::DistName(c.dist)
                    << " iter=" << iter
                    << " seed=" << seed
                    << " m=" << c.m << " n=" << c.n
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
        all << "ReluForward sweep failed in " << failures.size() << " case(s):\n";
        for (const auto& f : failures) {
            all << "  " << f << "\n";
        }
        ADD_FAILURE() << all.str();
    }
}

TEST(ReluBackward, SweepAllCases) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    const std::vector<fa_test::ReluCase> cases = fa_test::BuildBackwardCases();
    std::vector<std::string> failures;
    failures.reserve(64);

    for (const auto& c : cases) {
        for (int iter = 0; iter < c.iters; ++iter) {
            const uint32_t seed = fa_test::MixSeed(fa_test::kReluSeedBase, c.m, c.n, iter, 1);
            const std::vector<float> x_ref =
                fa_test::SampleUniformVector(static_cast<size_t>(c.m) * c.n, c.lo, c.hi, seed);
            const std::vector<float> dY_ref =
                fa_test::SampleUniformVector(static_cast<size_t>(c.m) * c.n, c.lo, c.hi, seed + 1u);

            Stream stream;
            Tensor x_h = MakeCpuTensor2D(c.m, c.n, x_ref);
            Tensor dY_h = MakeCpuTensor2D(c.m, c.n, dY_ref);
            Tensor x_d = x_h.clone(Device::CUDA, stream);
            Tensor dY_d = dY_h.clone(Device::CUDA, stream);

            ReluResults out = relu_forward(x_d, &stream);
            ReluGrads grads = relu_backward(dY_d, out.ctx, &stream);
            stream.synchronize();

            const std::vector<float> got = grads.dX.clone(Device::CPU).to_vector<float>();
            const std::vector<float> expected = fa_test::reference_relu_backward(dY_ref, x_ref);

            int fail_count = 0;
            float worst_abs_err = 0.0f;
            float worst_tol = 0.0f;
            int worst_idx = -1;
            for (size_t i = 0; i < got.size(); ++i) {
                const float abs_err = std::fabs(got[i] - expected[i]);
                const float tol = c.abs_tol + c.rel_tol * std::fabs(expected[i]);
                if (abs_err > tol) {
                    ++fail_count;
                    if (abs_err > worst_abs_err) {
                        worst_abs_err = abs_err;
                        worst_tol = tol;
                        worst_idx = static_cast<int>(i);
                    }
                }
            }

            if (fail_count > 0) {
                std::ostringstream one;
                one << "case=" << c.name
                    << " dist=" << fa_test::DistName(c.dist)
                    << " iter=" << iter
                    << " seed=" << seed
                    << " m=" << c.m << " n=" << c.n
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
        all << "ReluBackward sweep failed in " << failures.size() << " case(s):\n";
        for (const auto& f : failures) {
            all << "  " << f << "\n";
        }
        ADD_FAILURE() << all.str();
    }
}

TEST(ReluForward, NumericEdgePatternsAndShapes) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    std::vector<fa_test::ForwardEdgeCase> cases;

    cases.push_back(fa_test::ForwardEdgeCase{
        "zeros_11", 1, 1, 1e-7f, 1e-7f, {0.0f}});

    cases.push_back(fa_test::ForwardEdgeCase{
        "sign_mix_44", 4, 4, 1e-7f, 1e-7f,
        {-3.0f, -2.0f, -1.0f, -0.0f,
          0.0f,  1.0f,  2.0f,  3.0f,
         -1e-7f, 1e-7f, -5.0f, 5.0f,
         -8.0f,  8.0f, -9.0f, 9.0f}});

    {
        const int m = 3;
        const int n = 5;
        std::vector<float> x(static_cast<size_t>(m) * n);
        for (int i = 0; i < m * n; ++i) {
            const float big = (i % 2 == 0) ? 1.0e3f : -1.0e3f;
            const float tiny = (i % 3 == 0) ? 1.0e-3f : -1.0e-3f;
            x[static_cast<size_t>(i)] = big + tiny;
        }
        cases.push_back(fa_test::ForwardEdgeCase{"mixed_mag", m, n, 1e-5f, 1e-6f, x});
    }

    {
        const int m = 256;
        const int n = 8;
        const uint32_t seed = fa_test::MixSeed(fa_test::kReluSeedBase, m, n, 1, 2);
        std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed);
        cases.push_back(fa_test::ForwardEdgeCase{"tall_skinny", m, n, 1e-6f, 1e-6f, x});
    }

    {
        const int m = 8;
        const int n = 256;
        const uint32_t seed = fa_test::MixSeed(fa_test::kReluSeedBase, m, n, 2, 3);
        std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed);
        cases.push_back(fa_test::ForwardEdgeCase{"short_fat", m, n, 1e-6f, 1e-6f, x});
    }

    for (const auto& c : cases) {
        Stream stream;
        Tensor x_h = MakeCpuTensor2D(c.m, c.n, c.x);
        const std::vector<float> got = RunForwardToHost(x_h, stream);
        const std::vector<float> expected = fa_test::reference_relu_forward(c.x);
        const uint32_t seed = fa_test::MixSeed(fa_test::kReluSeedBase, c.m, c.n, 0, 0);
        ExpectVectorNear(got, expected, c.abs_tol, c.rel_tol, ReproTag(c.name, seed, c.m, c.n));
    }
}

TEST(ReluForward, InvariantNonNegativeOutput) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    constexpr int m = 97;
    constexpr int n = 131;
    const uint32_t seed = fa_test::MixSeed(fa_test::kReluSeedBase, m, n, 10, 0);

    const std::vector<float> x_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -100.0f, 100.0f, seed);

    Stream stream;
    Tensor x_h = MakeCpuTensor2D(m, n, x_ref);
    Tensor x_d = x_h.clone(Device::CUDA, stream);
    ReluResults out = relu_forward(x_d, &stream);
    Tensor y_h = out.Y.clone(Device::CPU);
    stream.synchronize();
    const std::vector<float> y = y_h.to_vector<float>();

    for (size_t i = 0; i < y.size(); ++i) {
        EXPECT_GE(y[i], 0.0f) << ReproTag("non_negative", seed, m, n) << " idx=" << i;
    }
}

TEST(ReluForward, InvariantIdempotence) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    constexpr int m = 23;
    constexpr int n = 41;
    constexpr float abs_tol = 1e-6f;
    constexpr float rel_tol = 1e-6f;
    const uint32_t seed = fa_test::MixSeed(fa_test::kReluSeedBase, m, n, 11, 0);

    const std::vector<float> x_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -2.0f, 2.0f, seed);

    Stream stream;
    Tensor x_h = MakeCpuTensor2D(m, n, x_ref);
    Tensor x_d = x_h.clone(Device::CUDA, stream);
    ReluResults out1 = relu_forward(x_d, &stream);
    ReluResults out2 = relu_forward(out1.Y, &stream);

    stream.synchronize();
    const std::vector<float> y1 = out1.Y.clone(Device::CPU).to_vector<float>();
    const std::vector<float> y2 = out2.Y.clone(Device::CPU).to_vector<float>();

    ExpectVectorNear(y2, y1, abs_tol, rel_tol, ReproTag("idempotence", seed, m, n));
}

TEST(ReluForward, InvariantPositiveHomogeneity) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    constexpr int m = 17;
    constexpr int n = 29;
    constexpr float a = 2.5f;
    constexpr float abs_tol = 1e-6f;
    constexpr float rel_tol = 1e-6f;
    const uint32_t seed = fa_test::MixSeed(fa_test::kReluSeedBase, m, n, 12, 0);

    const std::vector<float> x_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -5.0f, 5.0f, seed);
    std::vector<float> ax_ref = x_ref;
    for (float& v : ax_ref) {
        v *= a;
    }

    Stream stream;
    Tensor x_h = MakeCpuTensor2D(m, n, x_ref);
    Tensor ax_h = MakeCpuTensor2D(m, n, ax_ref);
    Tensor x_d = x_h.clone(Device::CUDA, stream);
    Tensor ax_d = ax_h.clone(Device::CUDA, stream);

    ReluResults y = relu_forward(x_d, &stream);
    ReluResults ay = relu_forward(ax_d, &stream);

    stream.synchronize();
    std::vector<float> y_vec = y.Y.clone(Device::CPU).to_vector<float>();
    std::vector<float> ay_vec = ay.Y.clone(Device::CPU).to_vector<float>();

    for (float& v : y_vec) {
        v *= a;
    }

    ExpectVectorNear(ay_vec, y_vec, abs_tol, rel_tol, ReproTag("positive_homogeneity", seed, m, n));
}

TEST(ReluForward, InvariantDeterministicForSameInput) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    constexpr int m = 31;
    constexpr int n = 47;
    const uint32_t seed = fa_test::MixSeed(fa_test::kReluSeedBase, m, n, 13, 0);

    const std::vector<float> x_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -3.0f, 3.0f, seed);

    Stream stream;
    Tensor x_h = MakeCpuTensor2D(m, n, x_ref);
    Tensor x_d = x_h.clone(Device::CUDA, stream);

    ReluResults y1 = relu_forward(x_d, &stream);
    ReluResults y2 = relu_forward(x_d, &stream);
    stream.synchronize();

    const std::vector<float> a = y1.Y.clone(Device::CPU).to_vector<float>();
    const std::vector<float> b = y2.Y.clone(Device::CPU).to_vector<float>();

    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_FLOAT_EQ(a[i], b[i]) << ReproTag("det_fwd", seed, m, n) << " idx=" << i;
    }
}

TEST(ReluBackward, FiniteDifferenceGradientCheckAwayFromZero) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 3;
    constexpr int n = 4;
    constexpr float eps = 1e-3f;
    constexpr float abs_tol = 3e-2f;
    constexpr float rel_tol = 2e-2f;

    const uint32_t seed = fa_test::MixSeed(fa_test::kReluSeedBase, m, n, 14, 0);
    std::vector<float> x =
        fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -0.8f, 0.8f, seed);
    for (float& v : x) {
        if (std::fabs(v) < 0.2f) {
            v = (v >= 0.0f) ? 0.2f : -0.2f;
        }
    }
    const std::vector<float> dY_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -0.6f, 0.6f, seed + 1u);

    Stream stream;
    Tensor x_h = MakeCpuTensor2D(m, n, x);
    Tensor dY_h = MakeCpuTensor2D(m, n, dY_ref);
    Tensor x_d = x_h.clone(Device::CUDA, stream);
    Tensor dY_d = dY_h.clone(Device::CUDA, stream);

    ReluResults out = relu_forward(x_d, &stream);
    ReluGrads grads = relu_backward(dY_d, out.ctx, &stream);
    stream.synchronize();
    const std::vector<float> dX = grads.dX.clone(Device::CPU).to_vector<float>();

    for (size_t i = 0; i < x.size(); ++i) {
        std::vector<float> x_plus = x;
        std::vector<float> x_minus = x;
        x_plus[i] += eps;
        x_minus[i] -= eps;

        const float lp = ForwardLossDotDY(x_plus, dY_ref, m, n, stream);
        const float lm = ForwardLossDotDY(x_minus, dY_ref, m, n, stream);
        const float g_num = (lp - lm) / (2.0f * eps);
        const float tol = abs_tol + rel_tol * std::fabs(g_num);
        EXPECT_NEAR(dX[i], g_num, tol) << ReproTag("fd_dx", seed, m, n) << " idx=" << i;
    }
}

TEST(ReluBackward, InvariantZeroDYGivesZeroDX) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 19;
    constexpr int n = 23;
    constexpr float zero_abs_tol = 1e-7f;
    const uint32_t seed = fa_test::MixSeed(fa_test::kReluSeedBase, m, n, 15, 0);

    const std::vector<float> x_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -2.0f, 2.0f, seed);
    const std::vector<float> dY_ref(static_cast<size_t>(m) * n, 0.0f);

    Stream stream;
    Tensor x_h = MakeCpuTensor2D(m, n, x_ref);
    Tensor dY_h = MakeCpuTensor2D(m, n, dY_ref);
    Tensor x_d = x_h.clone(Device::CUDA, stream);
    Tensor dY_d = dY_h.clone(Device::CUDA, stream);

    ReluResults out = relu_forward(x_d, &stream);
    ReluGrads grads = relu_backward(dY_d, out.ctx, &stream);
    stream.synchronize();

    const std::vector<float> dX = grads.dX.clone(Device::CPU).to_vector<float>();
    const std::vector<float> expected(static_cast<size_t>(m) * n, 0.0f);
    ExpectVectorNear(dX, expected, zero_abs_tol, 0.0f, ReproTag("zero_dy", seed, m, n));
}

TEST(ReluBackward, InvariantMaskingRule) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 4;
    constexpr int n = 6;
    const uint32_t seed = fa_test::MixSeed(fa_test::kReluSeedBase, m, n, 16, 0);

    const std::vector<float> x_ref = {
        -4.0f, -1.0f, -0.0f, 0.0f, 0.1f, 1.0f,
        -2.0f, -0.5f, 0.0f,  0.3f, 2.0f, 3.0f,
        -7.0f, -6.0f, -5.0f, 4.0f, 5.0f, 6.0f,
        -1e-6f, 1e-6f, -8.0f, 8.0f, -9.0f, 9.0f,
    };
    const std::vector<float> dY_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 1u);

    Stream stream;
    Tensor x_h = MakeCpuTensor2D(m, n, x_ref);
    Tensor dY_h = MakeCpuTensor2D(m, n, dY_ref);
    Tensor x_d = x_h.clone(Device::CUDA, stream);
    Tensor dY_d = dY_h.clone(Device::CUDA, stream);

    ReluResults out = relu_forward(x_d, &stream);
    ReluGrads grads = relu_backward(dY_d, out.ctx, &stream);
    stream.synchronize();

    const std::vector<float> dX = grads.dX.clone(Device::CPU).to_vector<float>();
    for (size_t i = 0; i < dX.size(); ++i) {
        const float expected = x_ref[i] > 0.0f ? dY_ref[i] : 0.0f;
        EXPECT_FLOAT_EQ(dX[i], expected) << ReproTag("masking", seed, m, n) << " idx=" << i;
    }
}

TEST(ReluBackward, InvariantLinearityInDYForFixedX) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 11;
    constexpr int n = 17;
    constexpr float a = 1.7f;
    constexpr float b = -0.4f;
    constexpr float abs_tol = 1e-6f;
    constexpr float rel_tol = 1e-6f;
    const uint32_t seed = fa_test::MixSeed(fa_test::kReluSeedBase, m, n, 17, 0);

    const std::vector<float> x_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -3.0f, 3.0f, seed);
    const std::vector<float> dY1 =
        fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 1u);
    const std::vector<float> dY2 =
        fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 2u);

    std::vector<float> dY_comb(dY1.size());
    for (size_t i = 0; i < dY1.size(); ++i) {
        dY_comb[i] = a * dY1[i] + b * dY2[i];
    }

    Stream stream;
    Tensor x_h = MakeCpuTensor2D(m, n, x_ref);
    Tensor dY1_h = MakeCpuTensor2D(m, n, dY1);
    Tensor dY2_h = MakeCpuTensor2D(m, n, dY2);
    Tensor dYc_h = MakeCpuTensor2D(m, n, dY_comb);

    Tensor x_d = x_h.clone(Device::CUDA, stream);
    Tensor dY1_d = dY1_h.clone(Device::CUDA, stream);
    Tensor dY2_d = dY2_h.clone(Device::CUDA, stream);
    Tensor dYc_d = dYc_h.clone(Device::CUDA, stream);

    ReluResults out = relu_forward(x_d, &stream);
    ReluGrads g1 = relu_backward(dY1_d, out.ctx, &stream);
    ReluGrads g2 = relu_backward(dY2_d, out.ctx, &stream);
    ReluGrads gc = relu_backward(dYc_d, out.ctx, &stream);
    stream.synchronize();

    const std::vector<float> g1v = g1.dX.clone(Device::CPU).to_vector<float>();
    const std::vector<float> g2v = g2.dX.clone(Device::CPU).to_vector<float>();
    const std::vector<float> gcv = gc.dX.clone(Device::CPU).to_vector<float>();

    std::vector<float> expected(g1v.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = a * g1v[i] + b * g2v[i];
    }

    ExpectVectorNear(gcv, expected, abs_tol, rel_tol, ReproTag("lin_dy", seed, m, n));
}

TEST(ReluBackward, InvariantDeterministicForSameCtxAndDY) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 29;
    constexpr int n = 35;
    const uint32_t seed = fa_test::MixSeed(fa_test::kReluSeedBase, m, n, 18, 0);

    const std::vector<float> x_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -3.0f, 3.0f, seed);
    const std::vector<float> dY_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -3.0f, 3.0f, seed + 1u);

    Stream stream;
    Tensor x_h = MakeCpuTensor2D(m, n, x_ref);
    Tensor dY_h = MakeCpuTensor2D(m, n, dY_ref);
    Tensor x_d = x_h.clone(Device::CUDA, stream);
    Tensor dY_d = dY_h.clone(Device::CUDA, stream);

    ReluResults out = relu_forward(x_d, &stream);
    ReluGrads g1 = relu_backward(dY_d, out.ctx, &stream);
    ReluGrads g2 = relu_backward(dY_d, out.ctx, &stream);
    stream.synchronize();

    const std::vector<float> a = g1.dX.clone(Device::CPU).to_vector<float>();
    const std::vector<float> b = g2.dX.clone(Device::CPU).to_vector<float>();

    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_FLOAT_EQ(a[i], b[i]) << ReproTag("det_bwd", seed, m, n) << " idx=" << i;
    }
}

TEST(ReluBackward, ZeroBoundaryConvention) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 3;
    constexpr int n = 4;
    const uint32_t seed = fa_test::MixSeed(fa_test::kReluSeedBase, m, n, 19, 0);

    const std::vector<float> x_ref = {
        -1.0f, 0.0f, 1.0f, 0.0f,
        -2.0f, 2.0f, 0.0f, 3.0f,
        0.0f, -4.0f, 4.0f, 0.0f,
    };
    const std::vector<float> dY_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 1u);

    Stream stream;
    Tensor x_h = MakeCpuTensor2D(m, n, x_ref);
    Tensor dY_h = MakeCpuTensor2D(m, n, dY_ref);
    Tensor x_d = x_h.clone(Device::CUDA, stream);
    Tensor dY_d = dY_h.clone(Device::CUDA, stream);

    ReluResults out = relu_forward(x_d, &stream);
    ReluGrads grads = relu_backward(dY_d, out.ctx, &stream);
    stream.synchronize();

    const std::vector<float> dX = grads.dX.clone(Device::CPU).to_vector<float>();
    for (size_t i = 0; i < x_ref.size(); ++i) {
        if (x_ref[i] == 0.0f) {
            EXPECT_FLOAT_EQ(dX[i], 0.0f) << ReproTag("zero_boundary", seed, m, n) << " idx=" << i;
        }
    }
}

TEST(ReluForwardBackward, CtxStoresXByPointer) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream stream;
    Tensor x({6, 8}, DType::F32, Device::CUDA, stream);
    ReluResults out = relu_forward(x, &stream);
    ASSERT_EQ(out.ctx.X, &x);
}

TEST(ReluForwardBackward, CtxPointerTracksMutatedX) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 7;
    constexpr int n = 9;
    constexpr float abs_tol = 1e-6f;
    constexpr float rel_tol = 1e-6f;
    const uint32_t seed = fa_test::MixSeed(fa_test::kReluSeedBase, m, n, 20, 0);

    const std::vector<float> x_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed);
    const std::vector<float> dY_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 1u);

    Stream stream;
    Tensor x_h = MakeCpuTensor2D(m, n, x_ref);
    Tensor dY_h = MakeCpuTensor2D(m, n, dY_ref);
    Tensor x_d = x_h.clone(Device::CUDA, stream);
    Tensor dY_d = dY_h.clone(Device::CUDA, stream);

    ReluResults out = relu_forward(x_d, &stream);
    ReluGrads g_before = relu_backward(dY_d, out.ctx, &stream);

    std::vector<float> x2_ref = x_ref;
    for (float& v : x2_ref) {
        v = v * -0.75f + 0.1f;
    }
    Tensor x2_h = MakeCpuTensor2D(m, n, x2_ref);
    x_d.copy_from(x2_h, stream);

    ReluGrads g_after = relu_backward(dY_d, out.ctx, &stream);
    stream.synchronize();

    const std::vector<float> before = g_before.dX.clone(Device::CPU).to_vector<float>();
    const std::vector<float> after = g_after.dX.clone(Device::CPU).to_vector<float>();

    const std::vector<float> expected_before = fa_test::reference_relu_backward(dY_ref, x_ref);
    const std::vector<float> expected_after = fa_test::reference_relu_backward(dY_ref, x2_ref);

    ExpectVectorNear(before, expected_before, abs_tol, rel_tol, ReproTag("ctx_mut_before", seed, m, n));
    ExpectVectorNear(after, expected_after, abs_tol, rel_tol, ReproTag("ctx_mut_after", seed, m, n));
}

TEST(ReluForwardBackward, TwoStageReuseNoMidTransfer) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 11;
    constexpr int n = 13;
    constexpr float abs_tol = 1e-6f;
    constexpr float rel_tol = 1e-6f;
    const uint32_t seed = fa_test::MixSeed(fa_test::kReluSeedBase, m, n, 21, 0);

    const std::vector<float> x_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed);
    const std::vector<float> dY1_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 1u);
    const std::vector<float> dY2_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 2u);

    Stream stream;

    Tensor x_d = MakeCpuTensor2D(m, n, x_ref).clone(Device::CUDA, stream);
    Tensor dY1_d = MakeCpuTensor2D(m, n, dY1_ref).clone(Device::CUDA, stream);
    Tensor dY2_d = MakeCpuTensor2D(m, n, dY2_ref).clone(Device::CUDA, stream);

    ReluResults out1 = relu_forward(x_d, &stream);
    ReluGrads g1 = relu_backward(dY1_d, out1.ctx, &stream);

    ApplyAffineInplaceF32(x_d, 0.5f, 0.125f, stream);

    ReluResults out2 = relu_forward(x_d, &stream);
    ReluGrads g2 = relu_backward(dY2_d, out2.ctx, &stream);

    stream.synchronize();
    const std::vector<float> g1_dx = g1.dX.clone(Device::CPU).to_vector<float>();
    const std::vector<float> g2_dx = g2.dX.clone(Device::CPU).to_vector<float>();

    std::vector<float> x2_ref = x_ref;
    for (float& v : x2_ref) {
        v = v * 0.5f + 0.125f;
    }

    const std::vector<float> g1_expected = fa_test::reference_relu_backward(dY1_ref, x_ref);
    const std::vector<float> g2_expected = fa_test::reference_relu_backward(dY2_ref, x2_ref);

    ExpectVectorNear(g1_dx, g1_expected, abs_tol, rel_tol, ReproTag("combined_2stage_s1", seed, m, n));
    ExpectVectorNear(g2_dx, g2_expected, abs_tol, rel_tol, ReproTag("combined_2stage_s2", seed, m, n));
}

TEST(ReluForwardBackward, CtxIsolationAcrossMultipleForwardsNoMidTransfer) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m1 = 5;
    constexpr int n1 = 7;
    constexpr int m2 = 6;
    constexpr int n2 = 4;
    constexpr float abs_tol = 1e-6f;
    constexpr float rel_tol = 1e-6f;

    const uint32_t seed1 = fa_test::MixSeed(fa_test::kReluSeedBase, m1, n1, 22, 0);
    const uint32_t seed2 = fa_test::MixSeed(fa_test::kReluSeedBase, m2, n2, 23, 0);

    const std::vector<float> x1_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m1) * n1, -1.0f, 1.0f, seed1);
    const std::vector<float> x2_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m2) * n2, -1.0f, 1.0f, seed2);
    const std::vector<float> dY1_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m1) * n1, -1.0f, 1.0f, seed1 + 1u);
    const std::vector<float> dY2_ref =
        fa_test::SampleUniformVector(static_cast<size_t>(m2) * n2, -1.0f, 1.0f, seed2 + 1u);

    Stream stream;

    Tensor x1_d = MakeCpuTensor2D(m1, n1, x1_ref).clone(Device::CUDA, stream);
    Tensor x2_d = MakeCpuTensor2D(m2, n2, x2_ref).clone(Device::CUDA, stream);
    Tensor dY1_d = MakeCpuTensor2D(m1, n1, dY1_ref).clone(Device::CUDA, stream);
    Tensor dY2_d = MakeCpuTensor2D(m2, n2, dY2_ref).clone(Device::CUDA, stream);

    ReluResults out1 = relu_forward(x1_d, &stream);
    ReluResults out2 = relu_forward(x2_d, &stream);

    ReluGrads g2 = relu_backward(dY2_d, out2.ctx, &stream);
    ReluGrads g1 = relu_backward(dY1_d, out1.ctx, &stream);

    stream.synchronize();
    const std::vector<float> got1 = g1.dX.clone(Device::CPU).to_vector<float>();
    const std::vector<float> got2 = g2.dX.clone(Device::CPU).to_vector<float>();

    const std::vector<float> expected1 = fa_test::reference_relu_backward(dY1_ref, x1_ref);
    const std::vector<float> expected2 = fa_test::reference_relu_backward(dY2_ref, x2_ref);

    ExpectVectorNear(got1, expected1, abs_tol, rel_tol, ReproTag("ctx_iso_1", seed1, m1, n1));
    ExpectVectorNear(got2, expected2, abs_tol, rel_tol, ReproTag("ctx_iso_2", seed2, m2, n2));
}

TEST(ReluForwardBackward, SweepAllCasesNoMidTransfer) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    const std::vector<fa_test::ReluCase> cases = fa_test::BuildForwardBackwardCases();
    std::vector<std::string> failures;

    for (const auto& c : cases) {
        for (int iter = 0; iter < c.iters; ++iter) {
            const uint32_t seed = fa_test::MixSeed(fa_test::kReluSeedBase, c.m, c.n, iter, 24);
            const std::vector<float> x_ref =
                fa_test::SampleUniformVector(static_cast<size_t>(c.m) * c.n, c.lo, c.hi, seed);
            const std::vector<float> dY1_ref =
                fa_test::SampleUniformVector(static_cast<size_t>(c.m) * c.n, c.lo, c.hi, seed + 1u);
            const std::vector<float> dY2_ref =
                fa_test::SampleUniformVector(static_cast<size_t>(c.m) * c.n, c.lo, c.hi, seed + 2u);

            Stream stream;
            Tensor x_d = MakeCpuTensor2D(c.m, c.n, x_ref).clone(Device::CUDA, stream);
            Tensor dY1_d = MakeCpuTensor2D(c.m, c.n, dY1_ref).clone(Device::CUDA, stream);
            Tensor dY2_d = MakeCpuTensor2D(c.m, c.n, dY2_ref).clone(Device::CUDA, stream);

            ReluResults out1 = relu_forward(x_d, &stream);
            ReluGrads g1 = relu_backward(dY1_d, out1.ctx, &stream);

            ApplyAffineInplaceF32(x_d, 0.5f, 0.125f, stream);

            ReluResults out2 = relu_forward(x_d, &stream);
            ReluGrads g2 = relu_backward(dY2_d, out2.ctx, &stream);

            stream.synchronize();
            const std::vector<float> g1_dx = g1.dX.clone(Device::CPU).to_vector<float>();
            const std::vector<float> g2_dx = g2.dX.clone(Device::CPU).to_vector<float>();

            std::vector<float> x2_ref = x_ref;
            for (float& v : x2_ref) {
                v = v * 0.5f + 0.125f;
            }

            const std::vector<float> g1_expected = fa_test::reference_relu_backward(dY1_ref, x_ref);
            const std::vector<float> g2_expected = fa_test::reference_relu_backward(dY2_ref, x2_ref);

            auto first_fail_idx = [&](const std::vector<float>& got,
                                      const std::vector<float>& expected) -> int {
                if (got.size() != expected.size()) {
                    return -2;
                }
                for (size_t i = 0; i < got.size(); ++i) {
                    const float tol = c.abs_tol + c.rel_tol * std::fabs(expected[i]);
                    if (std::fabs(got[i] - expected[i]) > tol) {
                        return static_cast<int>(i);
                    }
                }
                return -1;
            };

            const int s1_fail = first_fail_idx(g1_dx, g1_expected);
            const int s2_fail = first_fail_idx(g2_dx, g2_expected);

            if (s1_fail != -1 || s2_fail != -1) {
                std::ostringstream one;
                one << "case=" << c.name
                    << " dist=" << fa_test::DistName(c.dist)
                    << " iter=" << iter
                    << " seed=" << seed
                    << " m=" << c.m << " n=" << c.n
                    << " s1_fail_idx=" << s1_fail
                    << " s2_fail_idx=" << s2_fail;
                failures.push_back(one.str());
            }
        }
    }

    if (!failures.empty()) {
        std::ostringstream all;
        all << "ReluForwardBackward sweep failed in " << failures.size() << " case(s):\n";
        for (const auto& f : failures) {
            all << "  " << f << "\n";
        }
        ADD_FAILURE() << all.str();
    }
}

TEST(ReluForward, SingleStreamOrderingReuseStressFixedShape) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    const int launches = fa_test::LongStressEnabled() ? 1400 : 500;
    constexpr int m = 96;
    constexpr int n = 144;
    constexpr float abs_tol = 1e-6f;
    constexpr float rel_tol = 1e-6f;

    Stream stream;
    std::vector<fa_test::QueuedReluForwardJob> jobs;
    jobs.reserve(static_cast<size_t>(launches));

    for (int i = 0; i < launches; ++i) {
        const uint32_t seed = fa_test::MixSeed(fa_test::kReluSeedBase, m, n, i, 25);
        const std::vector<float> x_ref =
            fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed);

        Tensor x_d = MakeCpuTensor2D(m, n, x_ref).clone(Device::CUDA, stream);
        ReluResults out = relu_forward(x_d, &stream);

        jobs.push_back(fa_test::QueuedReluForwardJob{
            m,
            n,
            abs_tol,
            rel_tol,
            std::move(x_d),
            out.Y.clone(Device::CPU),
            fa_test::reference_relu_forward(x_ref),
        });
    }

    stream.synchronize();
    fa_test::ValidateQueuedForwardJobs(jobs);
}

TEST(ReluForward, SingleStreamOrderingReuseStressShapeCycleABC) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    const fa_test::ReluShapeCfg cfgs[3] = {
        {17, 29, -1.0f, 1.0f, 1e-6f, 1e-6f},
        {73, 41, -1e-3f, 1e-3f, 1e-7f, 1e-6f},
        {128, 96, -10.0f, 10.0f, 1e-5f, 1e-6f},
    };

    const int launches = fa_test::LongStressEnabled() ? 900 : 360;
    Stream stream;
    std::vector<fa_test::QueuedReluForwardJob> jobs;
    jobs.reserve(static_cast<size_t>(launches));

    for (int i = 0; i < launches; ++i) {
        const auto& c = cfgs[i % 3];
        const uint32_t seed = fa_test::MixSeed(fa_test::kReluSeedBase, c.m, c.n, i, 26);

        const std::vector<float> x_ref =
            fa_test::SampleUniformVector(static_cast<size_t>(c.m) * c.n, c.lo, c.hi, seed);
        Tensor x_d = MakeCpuTensor2D(c.m, c.n, x_ref).clone(Device::CUDA, stream);
        ReluResults out = relu_forward(x_d, &stream);

        jobs.push_back(fa_test::QueuedReluForwardJob{
            c.m,
            c.n,
            c.abs_tol,
            c.rel_tol,
            std::move(x_d),
            out.Y.clone(Device::CPU),
            fa_test::reference_relu_forward(x_ref),
        });
    }

    stream.synchronize();
    fa_test::ValidateQueuedForwardJobs(jobs);
}

TEST(ReluBackward, SingleStreamOrderingReuseStressFixedShape) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    const int launches = fa_test::LongStressEnabled() ? 1400 : 500;
    constexpr int m = 96;
    constexpr int n = 144;
    constexpr float abs_tol = 1e-6f;
    constexpr float rel_tol = 1e-6f;

    Stream stream;
    std::vector<fa_test::QueuedReluBackwardJob> jobs;
    jobs.reserve(static_cast<size_t>(launches));

    for (int i = 0; i < launches; ++i) {
        const uint32_t seed = fa_test::MixSeed(fa_test::kReluSeedBase, m, n, i, 27);
        const std::vector<float> x_ref =
            fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed);
        const std::vector<float> dY_ref =
            fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 1u);

        Tensor x_d = MakeCpuTensor2D(m, n, x_ref).clone(Device::CUDA, stream);
        Tensor dY_d = MakeCpuTensor2D(m, n, dY_ref).clone(Device::CUDA, stream);

        ReluResults out = relu_forward(x_d, &stream);
        ReluGrads g = relu_backward(dY_d, out.ctx, &stream);

        jobs.push_back(fa_test::QueuedReluBackwardJob{
            m,
            n,
            abs_tol,
            rel_tol,
            std::move(x_d),
            std::move(dY_d),
            g.dX.clone(Device::CPU),
            fa_test::reference_relu_backward(dY_ref, x_ref),
        });
    }

    stream.synchronize();
    fa_test::ValidateQueuedBackwardJobs(jobs);
}

TEST(ReluBackward, SingleStreamOrderingReuseStressShapeCycleABC) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    const fa_test::ReluShapeCfg cfgs[3] = {
        {17, 29, -1.0f, 1.0f, 1e-6f, 1e-6f},
        {73, 41, -1e-3f, 1e-3f, 1e-7f, 1e-6f},
        {128, 96, -10.0f, 10.0f, 1e-5f, 1e-6f},
    };

    const int launches = fa_test::LongStressEnabled() ? 900 : 360;
    Stream stream;
    std::vector<fa_test::QueuedReluBackwardJob> jobs;
    jobs.reserve(static_cast<size_t>(launches));

    for (int i = 0; i < launches; ++i) {
        const auto& c = cfgs[i % 3];
        const uint32_t seed = fa_test::MixSeed(fa_test::kReluSeedBase, c.m, c.n, i, 28);

        const std::vector<float> x_ref =
            fa_test::SampleUniformVector(static_cast<size_t>(c.m) * c.n, c.lo, c.hi, seed);
        const std::vector<float> dY_ref =
            fa_test::SampleUniformVector(static_cast<size_t>(c.m) * c.n, c.lo, c.hi, seed + 1u);

        Tensor x_d = MakeCpuTensor2D(c.m, c.n, x_ref).clone(Device::CUDA, stream);
        Tensor dY_d = MakeCpuTensor2D(c.m, c.n, dY_ref).clone(Device::CUDA, stream);

        ReluResults out = relu_forward(x_d, &stream);
        ReluGrads g = relu_backward(dY_d, out.ctx, &stream);

        jobs.push_back(fa_test::QueuedReluBackwardJob{
            c.m,
            c.n,
            c.abs_tol,
            c.rel_tol,
            std::move(x_d),
            std::move(dY_d),
            g.dX.clone(Device::CPU),
            fa_test::reference_relu_backward(dY_ref, x_ref),
        });
    }

    stream.synchronize();
    fa_test::ValidateQueuedBackwardJobs(jobs);
}

}  // namespace
