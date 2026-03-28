#include "general.h"
#include "test_layernorm.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
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

Tensor MakeCpuTensor1D(int n, const std::vector<float>& values) {
    Tensor t({n}, DType::F32, Device::CPU);
    t.copy_from(values);
    return t;
}

float Dot(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("test_layernorm.cu: Dot inputs must have the same size.");
    }
    double acc = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        acc += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return static_cast<float>(acc);
}

__global__ void affine_inplace_kernel(float* data, int n, float scale, float bias) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * scale + bias;
    }
}

void ApplyAffineInplaceF32(Tensor& t, float scale, float bias, Stream& stream) {
    if (t.dtype_ != DType::F32 || t.device_ != Device::CUDA) {
        throw std::invalid_argument("test_layernorm.cu: ApplyAffineInplaceF32 expects CUDA float tensor.");
    }
    const int n = static_cast<int>(t.numel());
    const int block = 256;
    const int grid = (n + block - 1) / block;
    affine_inplace_kernel<<<grid, block, 0, stream.s>>>(static_cast<float*>(t.data_), n, scale, bias);
    CUDA_CHECK(cudaGetLastError());
}

float ForwardLossDotDY(const std::vector<float>& x,
                       const std::vector<float>& gamma,
                       const std::vector<float>& beta,
                       const std::vector<float>& dY_ref,
                       int m,
                       int n,
                       float eps,
                       Stream& stream) {
    Tensor x_h = MakeCpuTensor2D(m, n, x);
    Tensor gamma_h = MakeCpuTensor1D(n, gamma);
    Tensor beta_h = MakeCpuTensor1D(n, beta);

    Tensor x_d = x_h.clone(Device::CUDA, stream);
    Tensor gamma_d = gamma_h.clone(Device::CUDA, stream);
    Tensor beta_d = beta_h.clone(Device::CUDA, stream);

    LayerNormResults out = layernorm_forward(x_d, gamma_d, beta_d, eps, &stream);
    std::vector<float> y = out.Y.clone(Device::CPU).to_vector<float>();
    stream.synchronize();

    return Dot(y, dY_ref);
}

TEST(LayerNormForward, RejectsNullStream) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream stream;
    Tensor x({4, 7}, DType::F32, Device::CUDA, stream);
    Tensor gamma({7}, DType::F32, Device::CUDA, stream);
    Tensor beta({7}, DType::F32, Device::CUDA, stream);

    EXPECT_THROW((void)layernorm_forward(x, gamma, beta, 1e-5f, nullptr), std::invalid_argument);
}

TEST(LayerNormForward, RejectsNonDefaultStream) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream stream;
    Tensor x({4, 7}, DType::F32, Device::CUDA, stream);
    Tensor gamma({7}, DType::F32, Device::CUDA, stream);
    Tensor beta({7}, DType::F32, Device::CUDA, stream);

    cudaStream_t raw_non_default = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&raw_non_default, cudaStreamNonBlocking));
    Stream non_default_stream;
    non_default_stream.s = raw_non_default;
    non_default_stream.owns_ = false;

    EXPECT_THROW((void)layernorm_forward(x, gamma, beta, 1e-5f, &non_default_stream), std::invalid_argument);
    CUDA_CHECK(cudaStreamDestroy(raw_non_default));
}

TEST(LayerNormForward, RejectsEpsNonPositive) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream stream;
    Tensor x({4, 7}, DType::F32, Device::CUDA, stream);
    Tensor gamma({7}, DType::F32, Device::CUDA, stream);
    Tensor beta({7}, DType::F32, Device::CUDA, stream);

    EXPECT_THROW((void)layernorm_forward(x, gamma, beta, 0.0f, &stream), std::invalid_argument);
    EXPECT_THROW((void)layernorm_forward(x, gamma, beta, -1e-5f, &stream), std::invalid_argument);
}

TEST(LayerNormForward, RejectsNonF32) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream stream;
    Tensor x_bad({4, 7}, DType::F16, Device::CUDA, stream);
    Tensor x_good({4, 7}, DType::F32, Device::CUDA, stream);
    Tensor gamma_bad({7}, DType::F16, Device::CUDA, stream);
    Tensor gamma_good({7}, DType::F32, Device::CUDA, stream);
    Tensor beta_bad({7}, DType::F16, Device::CUDA, stream);
    Tensor beta_good({7}, DType::F32, Device::CUDA, stream);

    EXPECT_THROW((void)layernorm_forward(x_bad, gamma_good, beta_good, 1e-5f, &stream), std::invalid_argument);
    EXPECT_THROW((void)layernorm_forward(x_good, gamma_bad, beta_good, 1e-5f, &stream), std::invalid_argument);
    EXPECT_THROW((void)layernorm_forward(x_good, gamma_good, beta_bad, 1e-5f, &stream), std::invalid_argument);
}

TEST(LayerNormForward, RejectsNonCudaInput) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream stream;
    Tensor x_cpu({4, 7}, DType::F32, Device::CPU);
    Tensor x_cuda({4, 7}, DType::F32, Device::CUDA, stream);
    Tensor gamma_cpu({7}, DType::F32, Device::CPU);
    Tensor gamma_cuda({7}, DType::F32, Device::CUDA, stream);
    Tensor beta_cpu({7}, DType::F32, Device::CPU);
    Tensor beta_cuda({7}, DType::F32, Device::CUDA, stream);

    EXPECT_THROW((void)layernorm_forward(x_cpu, gamma_cuda, beta_cuda, 1e-5f, &stream), std::invalid_argument);
    EXPECT_THROW((void)layernorm_forward(x_cuda, gamma_cpu, beta_cuda, 1e-5f, &stream), std::invalid_argument);
    EXPECT_THROW((void)layernorm_forward(x_cuda, gamma_cuda, beta_cpu, 1e-5f, &stream), std::invalid_argument);
}

TEST(LayerNormForward, RejectsShapeMismatch) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream stream;
    Tensor x_non2d({4, 7, 2}, DType::F32, Device::CUDA, stream);
    Tensor x_good({4, 7}, DType::F32, Device::CUDA, stream);
    Tensor gamma_non1d({1, 7}, DType::F32, Device::CUDA, stream);
    Tensor beta_non1d({1, 7}, DType::F32, Device::CUDA, stream);
    Tensor gamma_bad({8}, DType::F32, Device::CUDA, stream);
    Tensor beta_bad({8}, DType::F32, Device::CUDA, stream);
    Tensor gamma_good({7}, DType::F32, Device::CUDA, stream);
    Tensor beta_good({7}, DType::F32, Device::CUDA, stream);

    EXPECT_THROW((void)layernorm_forward(x_non2d, gamma_good, beta_good, 1e-5f, &stream), std::invalid_argument);
    EXPECT_THROW((void)layernorm_forward(x_good, gamma_non1d, beta_good, 1e-5f, &stream), std::invalid_argument);
    EXPECT_THROW((void)layernorm_forward(x_good, gamma_good, beta_non1d, 1e-5f, &stream), std::invalid_argument);
    EXPECT_THROW((void)layernorm_forward(x_good, gamma_bad, beta_good, 1e-5f, &stream), std::invalid_argument);
    EXPECT_THROW((void)layernorm_forward(x_good, gamma_good, beta_bad, 1e-5f, &stream), std::invalid_argument);
}

TEST(LayerNormBackward, RejectsNullStream) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream stream;
    Tensor x({4, 7}, DType::F32, Device::CUDA, stream);
    Tensor gamma({7}, DType::F32, Device::CUDA, stream);
    Tensor beta({7}, DType::F32, Device::CUDA, stream);
    LayerNormResults out = layernorm_forward(x, gamma, beta, 1e-5f, &stream);
    Tensor dY({4, 7}, DType::F32, Device::CUDA, stream);

    EXPECT_THROW((void)layernorm_backward(dY, out.ctx, true, true, true, nullptr), std::invalid_argument);
}

TEST(LayerNormBackward, RejectsNonDefaultStream) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream stream;
    Tensor x({4, 7}, DType::F32, Device::CUDA, stream);
    Tensor gamma({7}, DType::F32, Device::CUDA, stream);
    Tensor beta({7}, DType::F32, Device::CUDA, stream);
    LayerNormResults out = layernorm_forward(x, gamma, beta, 1e-5f, &stream);
    Tensor dY({4, 7}, DType::F32, Device::CUDA, stream);

    cudaStream_t raw_non_default = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&raw_non_default, cudaStreamNonBlocking));
    Stream non_default_stream;
    non_default_stream.s = raw_non_default;
    non_default_stream.owns_ = false;

    EXPECT_THROW((void)layernorm_backward(dY, out.ctx, true, true, true, &non_default_stream), std::invalid_argument);
    CUDA_CHECK(cudaStreamDestroy(raw_non_default));
}

TEST(LayerNormBackward, RejectsNullCtxPointers) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream stream;
    Tensor x({4, 7}, DType::F32, Device::CUDA, stream);
    Tensor gamma({7}, DType::F32, Device::CUDA, stream);
    Tensor beta({7}, DType::F32, Device::CUDA, stream);
    LayerNormResults out = layernorm_forward(x, gamma, beta, 1e-5f, &stream);
    Tensor dY({4, 7}, DType::F32, Device::CUDA, stream);

    LayerNormCtx bad_x{nullptr, out.ctx.gamma, std::move(out.ctx.mean), std::move(out.ctx.rstd), out.ctx.eps, out.ctx.m, out.ctx.n};
    EXPECT_THROW((void)layernorm_backward(dY, bad_x, true, true, true, &stream), std::invalid_argument);

    LayerNormResults out2 = layernorm_forward(x, gamma, beta, 1e-5f, &stream);
    LayerNormCtx bad_gamma{out2.ctx.X, nullptr, std::move(out2.ctx.mean), std::move(out2.ctx.rstd), out2.ctx.eps, out2.ctx.m, out2.ctx.n};
    EXPECT_THROW((void)layernorm_backward(dY, bad_gamma, true, true, true, &stream), std::invalid_argument);
}

TEST(LayerNormBackward, RejectsInvalidCtxAndInputShapes) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream stream;
    Tensor x({4, 7}, DType::F32, Device::CUDA, stream);
    Tensor gamma({7}, DType::F32, Device::CUDA, stream);
    Tensor beta({7}, DType::F32, Device::CUDA, stream);
    LayerNormResults out = layernorm_forward(x, gamma, beta, 1e-5f, &stream);

    Tensor dY_non2d({4, 7, 1}, DType::F32, Device::CUDA, stream);
    Tensor dY_bad_shape({7, 4}, DType::F32, Device::CUDA, stream);
    Tensor dY_good({4, 7}, DType::F32, Device::CUDA, stream);
    Tensor dY_cpu({4, 7}, DType::F32, Device::CPU);
    Tensor dY_f16({4, 7}, DType::F16, Device::CUDA, stream);

    EXPECT_THROW((void)layernorm_backward(dY_non2d, out.ctx, true, true, true, &stream), std::invalid_argument);
    EXPECT_THROW((void)layernorm_backward(dY_bad_shape, out.ctx, true, true, true, &stream), std::invalid_argument);
    EXPECT_THROW((void)layernorm_backward(dY_cpu, out.ctx, true, true, true, &stream), std::invalid_argument);
    EXPECT_THROW((void)layernorm_backward(dY_f16, out.ctx, true, true, true, &stream), std::invalid_argument);

    LayerNormResults out2 = layernorm_forward(x, gamma, beta, 1e-5f, &stream);
    out2.ctx.eps = 0.0f;
    EXPECT_THROW((void)layernorm_backward(dY_good, out2.ctx, true, true, true, &stream), std::invalid_argument);

    LayerNormResults out3 = layernorm_forward(x, gamma, beta, 1e-5f, &stream);
    out3.ctx.m += 1;
    EXPECT_THROW((void)layernorm_backward(dY_good, out3.ctx, true, true, true, &stream), std::invalid_argument);

    LayerNormResults out4 = layernorm_forward(x, gamma, beta, 1e-5f, &stream);
    out4.ctx.n += 1;
    EXPECT_THROW((void)layernorm_backward(dY_good, out4.ctx, true, true, true, &stream), std::invalid_argument);

    Tensor mean_bad({5}, DType::F32, Device::CUDA, stream);
    Tensor rstd_good({4}, DType::F32, Device::CUDA, stream);
    LayerNormCtx bad_mean{&x, &gamma, std::move(mean_bad), std::move(rstd_good), 1e-5f, 4, 7};
    EXPECT_THROW((void)layernorm_backward(dY_good, bad_mean, true, true, true, &stream), std::invalid_argument);

    Tensor mean_good({4}, DType::F32, Device::CUDA, stream);
    Tensor rstd_bad({5}, DType::F32, Device::CUDA, stream);
    LayerNormCtx bad_rstd{&x, &gamma, std::move(mean_good), std::move(rstd_bad), 1e-5f, 4, 7};
    EXPECT_THROW((void)layernorm_backward(dY_good, bad_rstd, true, true, true, &stream), std::invalid_argument);
}

TEST(LayerNormForward, MatchesReferenceOddShape) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 13;
    constexpr int n = 37;
    constexpr float eps = 1e-5f;
    constexpr float abs_tol = 2e-4f;
    constexpr float rel_tol = 2e-4f;

    const uint32_t seed = fa_test::MixSeed(fa_test::kLayerNormSeedBase, m, n, 101, 0);
    const std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed);
    const std::vector<float> gamma = fa_test::SampleUniformVector(static_cast<size_t>(n), -1.0f, 1.0f, seed + 1u);
    const std::vector<float> beta = fa_test::SampleUniformVector(static_cast<size_t>(n), -1.0f, 1.0f, seed + 2u);

    Stream stream;
    Tensor x_d = MakeCpuTensor2D(m, n, x).clone(Device::CUDA, stream);
    Tensor gamma_d = MakeCpuTensor1D(n, gamma).clone(Device::CUDA, stream);
    Tensor beta_d = MakeCpuTensor1D(n, beta).clone(Device::CUDA, stream);

    LayerNormResults out = layernorm_forward(x_d, gamma_d, beta_d, eps, &stream);
    std::vector<float> y = out.Y.clone(Device::CPU).to_vector<float>();
    std::vector<float> mean = out.ctx.mean.clone(Device::CPU).to_vector<float>();
    std::vector<float> rstd = out.ctx.rstd.clone(Device::CPU).to_vector<float>();
    stream.synchronize();

    const fa_test::LayerNormRefForward ref = fa_test::reference_layernorm_forward(x, gamma, beta, m, n, eps);

    ExpectVectorNear(y, ref.y, abs_tol, rel_tol, ReproTag("forward_y", seed, m, n));
    ExpectVectorNear(mean, ref.mean, abs_tol, rel_tol, ReproTag("forward_mean", seed, m, n));
    ExpectVectorNear(rstd, ref.rstd, abs_tol, rel_tol, ReproTag("forward_rstd", seed, m, n));
}

TEST(LayerNormBackward, MatchesReferenceAllGradsOddShape) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 13;
    constexpr int n = 37;
    constexpr float eps = 1e-5f;
    constexpr float abs_tol = 5e-4f;
    constexpr float rel_tol = 5e-4f;

    const uint32_t seed = fa_test::MixSeed(fa_test::kLayerNormSeedBase, m, n, 201, 0);
    const std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed);
    const std::vector<float> gamma = fa_test::SampleUniformVector(static_cast<size_t>(n), -1.0f, 1.0f, seed + 1u);
    const std::vector<float> beta = fa_test::SampleUniformVector(static_cast<size_t>(n), -1.0f, 1.0f, seed + 2u);
    const std::vector<float> dY = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 3u);

    Stream stream;
    Tensor x_d = MakeCpuTensor2D(m, n, x).clone(Device::CUDA, stream);
    Tensor gamma_d = MakeCpuTensor1D(n, gamma).clone(Device::CUDA, stream);
    Tensor beta_d = MakeCpuTensor1D(n, beta).clone(Device::CUDA, stream);
    Tensor dY_d = MakeCpuTensor2D(m, n, dY).clone(Device::CUDA, stream);

    LayerNormResults out = layernorm_forward(x_d, gamma_d, beta_d, eps, &stream);
    LayerNormGrads grads = layernorm_backward(dY_d, out.ctx, true, true, true, &stream);

    ASSERT_TRUE(grads.has_dX);
    ASSERT_TRUE(grads.has_dgamma);
    ASSERT_TRUE(grads.has_dbeta);
    ASSERT_TRUE(grads.dX.has_value());
    ASSERT_TRUE(grads.dgamma.has_value());
    ASSERT_TRUE(grads.dbeta.has_value());

    std::vector<float> dX = grads.dX->clone(Device::CPU).to_vector<float>();
    std::vector<float> dgamma = grads.dgamma->clone(Device::CPU).to_vector<float>();
    std::vector<float> dbeta = grads.dbeta->clone(Device::CPU).to_vector<float>();
    std::vector<float> mean = out.ctx.mean.clone(Device::CPU).to_vector<float>();
    std::vector<float> rstd = out.ctx.rstd.clone(Device::CPU).to_vector<float>();
    stream.synchronize();

    const fa_test::LayerNormRefBackward ref =
        fa_test::reference_layernorm_backward(dY, x, gamma, mean, rstd, m, n, true, true, true);

    ExpectVectorNear(dX, ref.dX, abs_tol, rel_tol, ReproTag("backward_dx", seed, m, n));
    ExpectVectorNear(dgamma, ref.dgamma, abs_tol, rel_tol, ReproTag("backward_dgamma", seed, m, n));
    ExpectVectorNear(dbeta, ref.dbeta, abs_tol, rel_tol, ReproTag("backward_dbeta", seed, m, n));
}

TEST(LayerNormBackward, NeedsGradFlagsMatrix) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 8;
    constexpr int n = 6;
    constexpr float eps = 1e-5f;
    constexpr float abs_tol = 5e-4f;
    constexpr float rel_tol = 5e-4f;

    const uint32_t seed = fa_test::MixSeed(fa_test::kLayerNormSeedBase, m, n, 210, 0);
    const std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed);
    const std::vector<float> gamma = fa_test::SampleUniformVector(static_cast<size_t>(n), -1.0f, 1.0f, seed + 1u);
    const std::vector<float> beta = fa_test::SampleUniformVector(static_cast<size_t>(n), -1.0f, 1.0f, seed + 2u);
    const std::vector<float> dY = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 3u);

    Stream stream;
    Tensor x_d = MakeCpuTensor2D(m, n, x).clone(Device::CUDA, stream);
    Tensor gamma_d = MakeCpuTensor1D(n, gamma).clone(Device::CUDA, stream);
    Tensor beta_d = MakeCpuTensor1D(n, beta).clone(Device::CUDA, stream);
    Tensor dY_d = MakeCpuTensor2D(m, n, dY).clone(Device::CUDA, stream);

    LayerNormResults out = layernorm_forward(x_d, gamma_d, beta_d, eps, &stream);
    std::vector<float> mean = out.ctx.mean.clone(Device::CPU).to_vector<float>();
    std::vector<float> rstd = out.ctx.rstd.clone(Device::CPU).to_vector<float>();

    const bool flag_vals[2] = {false, true};
    for (bool need_dx : flag_vals) {
        for (bool need_dgamma : flag_vals) {
            for (bool need_dbeta : flag_vals) {
                LayerNormGrads grads = layernorm_backward(dY_d, out.ctx, need_dx, need_dgamma, need_dbeta, &stream);

                EXPECT_EQ(grads.has_dX, need_dx);
                EXPECT_EQ(grads.has_dgamma, need_dgamma);
                EXPECT_EQ(grads.has_dbeta, need_dbeta);
                EXPECT_EQ(grads.dX.has_value(), need_dx);
                EXPECT_EQ(grads.dgamma.has_value(), need_dgamma);
                EXPECT_EQ(grads.dbeta.has_value(), need_dbeta);

                const fa_test::LayerNormRefBackward ref =
                    fa_test::reference_layernorm_backward(dY, x, gamma, mean, rstd, m, n,
                                                          need_dx, need_dgamma, need_dbeta);

                if (need_dx) {
                    ExpectVectorNear(grads.dX->clone(Device::CPU).to_vector<float>(), ref.dX,
                                     abs_tol, rel_tol, ReproTag("flags_dx", seed, m, n));
                }
                if (need_dgamma) {
                    ExpectVectorNear(grads.dgamma->clone(Device::CPU).to_vector<float>(), ref.dgamma,
                                     abs_tol, rel_tol, ReproTag("flags_dgamma", seed, m, n));
                }
                if (need_dbeta) {
                    ExpectVectorNear(grads.dbeta->clone(Device::CPU).to_vector<float>(), ref.dbeta,
                                     abs_tol, rel_tol, ReproTag("flags_dbeta", seed, m, n));
                }
            }
        }
    }
}

TEST(LayerNormForward, SweepAllCases) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    constexpr float eps = 1e-5f;
    const std::vector<fa_test::LayerNormCase> cases = fa_test::BuildForwardCases();
    std::vector<std::string> failures;
    failures.reserve(64);

    for (const auto& c : cases) {
        for (int iter = 0; iter < c.iters; ++iter) {
            const uint32_t seed = fa_test::MixSeed(fa_test::kLayerNormSeedBase, c.m, c.n, iter, 1);
            const std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(c.m) * c.n, c.lo, c.hi, seed);
            const std::vector<float> gamma = fa_test::SampleUniformVector(static_cast<size_t>(c.n), c.lo, c.hi, seed + 1u);
            const std::vector<float> beta = fa_test::SampleUniformVector(static_cast<size_t>(c.n), c.lo, c.hi, seed + 2u);

            Stream stream;
            Tensor x_d = MakeCpuTensor2D(c.m, c.n, x).clone(Device::CUDA, stream);
            Tensor gamma_d = MakeCpuTensor1D(c.n, gamma).clone(Device::CUDA, stream);
            Tensor beta_d = MakeCpuTensor1D(c.n, beta).clone(Device::CUDA, stream);

            LayerNormResults out = layernorm_forward(x_d, gamma_d, beta_d, eps, &stream);
            std::vector<float> y = out.Y.clone(Device::CPU).to_vector<float>();
            stream.synchronize();

            const fa_test::LayerNormRefForward ref =
                fa_test::reference_layernorm_forward(x, gamma, beta, c.m, c.n, eps);

            int fail_count = 0;
            float worst_abs_err = 0.0f;
            float worst_tol = 0.0f;
            int worst_idx = -1;
            for (size_t i = 0; i < y.size(); ++i) {
                const float abs_err = std::fabs(y[i] - ref.y[i]);
                const float tol = c.abs_tol + c.rel_tol * std::fabs(ref.y[i]);
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
        all << "LayerNormForward sweep failed in " << failures.size() << " case(s):\n";
        for (const auto& f : failures) {
            all << "  " << f << "\n";
        }
        ADD_FAILURE() << all.str();
    }
}

TEST(LayerNormBackward, SweepAllCases) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    constexpr float eps = 1e-5f;
    const std::vector<fa_test::LayerNormCase> cases = fa_test::BuildBackwardCases();
    std::vector<std::string> failures;
    failures.reserve(64);

    for (const auto& c : cases) {
        for (int iter = 0; iter < c.iters; ++iter) {
            const uint32_t seed = fa_test::MixSeed(fa_test::kLayerNormSeedBase, c.m, c.n, iter, 2);
            const std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(c.m) * c.n, c.lo, c.hi, seed);
            const std::vector<float> gamma = fa_test::SampleUniformVector(static_cast<size_t>(c.n), c.lo, c.hi, seed + 1u);
            const std::vector<float> beta = fa_test::SampleUniformVector(static_cast<size_t>(c.n), c.lo, c.hi, seed + 2u);
            const std::vector<float> dY = fa_test::SampleUniformVector(static_cast<size_t>(c.m) * c.n, c.lo, c.hi, seed + 3u);

            Stream stream;
            Tensor x_d = MakeCpuTensor2D(c.m, c.n, x).clone(Device::CUDA, stream);
            Tensor gamma_d = MakeCpuTensor1D(c.n, gamma).clone(Device::CUDA, stream);
            Tensor beta_d = MakeCpuTensor1D(c.n, beta).clone(Device::CUDA, stream);
            Tensor dY_d = MakeCpuTensor2D(c.m, c.n, dY).clone(Device::CUDA, stream);

            LayerNormResults out = layernorm_forward(x_d, gamma_d, beta_d, eps, &stream);
            LayerNormGrads grads = layernorm_backward(dY_d, out.ctx, true, true, true, &stream);

            std::vector<float> dX = grads.dX->clone(Device::CPU).to_vector<float>();
            std::vector<float> dgamma = grads.dgamma->clone(Device::CPU).to_vector<float>();
            std::vector<float> dbeta = grads.dbeta->clone(Device::CPU).to_vector<float>();
            std::vector<float> mean = out.ctx.mean.clone(Device::CPU).to_vector<float>();
            std::vector<float> rstd = out.ctx.rstd.clone(Device::CPU).to_vector<float>();
            stream.synchronize();

            const fa_test::LayerNormRefBackward ref =
                fa_test::reference_layernorm_backward(dY, x, gamma, mean, rstd, c.m, c.n, true, true, true);

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

            const int dx_fail = first_fail_idx(dX, ref.dX);
            const int dgamma_fail = first_fail_idx(dgamma, ref.dgamma);
            const int dbeta_fail = first_fail_idx(dbeta, ref.dbeta);
            if (dx_fail != -1 || dgamma_fail != -1 || dbeta_fail != -1) {
                std::ostringstream one;
                one << "case=" << c.name
                    << " dist=" << fa_test::DistName(c.dist)
                    << " iter=" << iter
                    << " seed=" << seed
                    << " m=" << c.m << " n=" << c.n
                    << " dx_fail_idx=" << dx_fail
                    << " dgamma_fail_idx=" << dgamma_fail
                    << " dbeta_fail_idx=" << dbeta_fail;
                failures.push_back(one.str());
            }
        }
    }

    if (!failures.empty()) {
        std::ostringstream all;
        all << "LayerNormBackward sweep failed in " << failures.size() << " case(s):\n";
        for (const auto& f : failures) {
            all << "  " << f << "\n";
        }
        ADD_FAILURE() << all.str();
    }
}

TEST(LayerNormForward, NumericEdgePatternsAndShapes) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    constexpr float eps = 1e-5f;
    std::vector<fa_test::LayerNormEdgeCase> cases;

    cases.push_back(fa_test::LayerNormEdgeCase{
        "zeros_11", 1, 1, 1e-6f, 1e-6f,
        {0.0f},
        {1.0f},
        {0.0f}});

    cases.push_back(fa_test::LayerNormEdgeCase{
        "sign_mix_44", 4, 4, 5e-4f, 5e-4f,
        {-3.0f, -2.0f, -1.0f, -0.0f,
          0.0f,  1.0f,  2.0f,  3.0f,
         -1e-7f, 1e-7f, -5.0f, 5.0f,
         -8.0f,  8.0f, -9.0f, 9.0f},
        {1.0f, -0.5f, 0.25f, 2.0f},
        {0.1f, -0.2f, 0.3f, -0.4f}});

    {
        const int m = 3;
        const int n = 5;
        std::vector<float> x(static_cast<size_t>(m) * n);
        std::vector<float> gamma(static_cast<size_t>(n));
        std::vector<float> beta(static_cast<size_t>(n));
        for (int i = 0; i < m * n; ++i) {
            const float big = (i % 2 == 0) ? 1.0e3f : -1.0e3f;
            const float tiny = (i % 3 == 0) ? 1.0e-3f : -1.0e-3f;
            x[static_cast<size_t>(i)] = big + tiny;
        }
        for (int j = 0; j < n; ++j) {
            gamma[static_cast<size_t>(j)] = (j % 2 == 0) ? 1.25f : -0.75f;
            beta[static_cast<size_t>(j)] = 0.1f * static_cast<float>(j - 2);
        }
        cases.push_back(fa_test::LayerNormEdgeCase{"mixed_mag", m, n, 2e-3f, 2e-3f, x, gamma, beta});
    }

    {
        const int m = 256;
        const int n = 8;
        const uint32_t seed = fa_test::MixSeed(fa_test::kLayerNormSeedBase, m, n, 1, 3);
        std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed);
        std::vector<float> gamma = fa_test::SampleUniformVector(static_cast<size_t>(n), -1.0f, 1.0f, seed + 1u);
        std::vector<float> beta = fa_test::SampleUniformVector(static_cast<size_t>(n), -1.0f, 1.0f, seed + 2u);
        cases.push_back(fa_test::LayerNormEdgeCase{"tall_skinny", m, n, 4e-4f, 4e-4f, x, gamma, beta});
    }

    {
        const int m = 8;
        const int n = 256;
        const uint32_t seed = fa_test::MixSeed(fa_test::kLayerNormSeedBase, m, n, 2, 4);
        std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed);
        std::vector<float> gamma = fa_test::SampleUniformVector(static_cast<size_t>(n), -1.0f, 1.0f, seed + 1u);
        std::vector<float> beta = fa_test::SampleUniformVector(static_cast<size_t>(n), -1.0f, 1.0f, seed + 2u);
        cases.push_back(fa_test::LayerNormEdgeCase{"short_fat", m, n, 4e-4f, 4e-4f, x, gamma, beta});
    }

    for (const auto& c : cases) {
        Stream stream;
        Tensor x_d = MakeCpuTensor2D(c.m, c.n, c.x).clone(Device::CUDA, stream);
        Tensor gamma_d = MakeCpuTensor1D(c.n, c.gamma).clone(Device::CUDA, stream);
        Tensor beta_d = MakeCpuTensor1D(c.n, c.beta).clone(Device::CUDA, stream);

        LayerNormResults out = layernorm_forward(x_d, gamma_d, beta_d, eps, &stream);
        std::vector<float> y = out.Y.clone(Device::CPU).to_vector<float>();
        stream.synchronize();

        const fa_test::LayerNormRefForward ref =
            fa_test::reference_layernorm_forward(c.x, c.gamma, c.beta, c.m, c.n, eps);
        const uint32_t seed = fa_test::MixSeed(fa_test::kLayerNormSeedBase, c.m, c.n, 0, 0);
        ExpectVectorNear(y, ref.y, c.abs_tol, c.rel_tol, ReproTag(c.name, seed, c.m, c.n));
    }
}

TEST(LayerNormForward, InvariantIdentityAffineAndRowwiseNormalization) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    constexpr int m = 23;
    constexpr int n = 41;
    constexpr float eps = 1e-5f;
    constexpr float abs_tol = 5e-4f;
    constexpr float rel_tol = 5e-4f;
    const uint32_t seed = fa_test::MixSeed(fa_test::kLayerNormSeedBase, m, n, 300, 0);

    const std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -2.0f, 2.0f, seed);
    const std::vector<float> gamma(static_cast<size_t>(n), 1.0f);
    const std::vector<float> beta(static_cast<size_t>(n), 0.0f);

    Stream stream;
    Tensor x_d = MakeCpuTensor2D(m, n, x).clone(Device::CUDA, stream);
    Tensor gamma_d = MakeCpuTensor1D(n, gamma).clone(Device::CUDA, stream);
    Tensor beta_d = MakeCpuTensor1D(n, beta).clone(Device::CUDA, stream);

    LayerNormResults out = layernorm_forward(x_d, gamma_d, beta_d, eps, &stream);
    std::vector<float> y = out.Y.clone(Device::CPU).to_vector<float>();
    stream.synchronize();

    const fa_test::LayerNormRefForward ref =
        fa_test::reference_layernorm_forward(x, gamma, beta, m, n, eps);

    ExpectVectorNear(y, ref.xhat, abs_tol, rel_tol, ReproTag("identity_affine", seed, m, n));

    for (int row = 0; row < m; ++row) {
        double sum = 0.0;
        double sq_sum = 0.0;
        for (int col = 0; col < n; ++col) {
            const double v = static_cast<double>(y[static_cast<size_t>(row) * n + col]);
            sum += v;
            sq_sum += v * v;
        }
        const double mean = sum / static_cast<double>(n);
        const double var = sq_sum / static_cast<double>(n) - mean * mean;
        EXPECT_NEAR(mean, 0.0, 1e-3) << ReproTag("row_mean", seed, m, n) << " row=" << row;
        EXPECT_NEAR(var, 1.0, 2e-3) << ReproTag("row_var", seed, m, n) << " row=" << row;
    }
}

TEST(LayerNormForward, InvariantDeterministicForSameInput) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    constexpr int m = 31;
    constexpr int n = 47;
    constexpr float eps = 1e-5f;
    const uint32_t seed = fa_test::MixSeed(fa_test::kLayerNormSeedBase, m, n, 301, 0);

    const std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -3.0f, 3.0f, seed);
    const std::vector<float> gamma = fa_test::SampleUniformVector(static_cast<size_t>(n), -3.0f, 3.0f, seed + 1u);
    const std::vector<float> beta = fa_test::SampleUniformVector(static_cast<size_t>(n), -3.0f, 3.0f, seed + 2u);

    Stream stream;
    Tensor x_d = MakeCpuTensor2D(m, n, x).clone(Device::CUDA, stream);
    Tensor gamma_d = MakeCpuTensor1D(n, gamma).clone(Device::CUDA, stream);
    Tensor beta_d = MakeCpuTensor1D(n, beta).clone(Device::CUDA, stream);

    LayerNormResults out1 = layernorm_forward(x_d, gamma_d, beta_d, eps, &stream);
    LayerNormResults out2 = layernorm_forward(x_d, gamma_d, beta_d, eps, &stream);
    stream.synchronize();

    const std::vector<float> y1 = out1.Y.clone(Device::CPU).to_vector<float>();
    const std::vector<float> y2 = out2.Y.clone(Device::CPU).to_vector<float>();
    const std::vector<float> m1 = out1.ctx.mean.clone(Device::CPU).to_vector<float>();
    const std::vector<float> m2 = out2.ctx.mean.clone(Device::CPU).to_vector<float>();
    const std::vector<float> r1 = out1.ctx.rstd.clone(Device::CPU).to_vector<float>();
    const std::vector<float> r2 = out2.ctx.rstd.clone(Device::CPU).to_vector<float>();

    ASSERT_EQ(y1.size(), y2.size());
    ASSERT_EQ(m1.size(), m2.size());
    ASSERT_EQ(r1.size(), r2.size());
    for (size_t i = 0; i < y1.size(); ++i) {
        EXPECT_FLOAT_EQ(y1[i], y2[i]) << ReproTag("det_y", seed, m, n) << " idx=" << i;
    }
    for (size_t i = 0; i < m1.size(); ++i) {
        EXPECT_FLOAT_EQ(m1[i], m2[i]) << ReproTag("det_mean", seed, m, n) << " idx=" << i;
        EXPECT_FLOAT_EQ(r1[i], r2[i]) << ReproTag("det_rstd", seed, m, n) << " idx=" << i;
    }
}

TEST(LayerNormBackward, FiniteDifferenceGradientCheckSmall) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 3;
    constexpr int n = 4;
    constexpr float eps = 1e-5f;
    constexpr float fd_eps = 1e-3f;
    constexpr float abs_tol = 3e-2f;
    constexpr float rel_tol = 2e-2f;

    const uint32_t seed = fa_test::MixSeed(fa_test::kLayerNormSeedBase, m, n, 400, 0);
    std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -0.8f, 0.8f, seed);
    const std::vector<float> gamma = fa_test::SampleUniformVector(static_cast<size_t>(n), -0.8f, 0.8f, seed + 1u);
    const std::vector<float> beta = fa_test::SampleUniformVector(static_cast<size_t>(n), -0.8f, 0.8f, seed + 2u);
    const std::vector<float> dY = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -0.6f, 0.6f, seed + 3u);

    Stream stream;
    Tensor x_d = MakeCpuTensor2D(m, n, x).clone(Device::CUDA, stream);
    Tensor gamma_d = MakeCpuTensor1D(n, gamma).clone(Device::CUDA, stream);
    Tensor beta_d = MakeCpuTensor1D(n, beta).clone(Device::CUDA, stream);
    Tensor dY_d = MakeCpuTensor2D(m, n, dY).clone(Device::CUDA, stream);

    LayerNormResults out = layernorm_forward(x_d, gamma_d, beta_d, eps, &stream);
    LayerNormGrads grads = layernorm_backward(dY_d, out.ctx, true, true, true, &stream);
    stream.synchronize();

    const std::vector<float> dX = grads.dX->clone(Device::CPU).to_vector<float>();
    const std::vector<float> dgamma = grads.dgamma->clone(Device::CPU).to_vector<float>();
    const std::vector<float> dbeta = grads.dbeta->clone(Device::CPU).to_vector<float>();

    for (size_t i = 0; i < x.size(); ++i) {
        std::vector<float> x_plus = x;
        std::vector<float> x_minus = x;
        x_plus[i] += fd_eps;
        x_minus[i] -= fd_eps;

        const float lp = ForwardLossDotDY(x_plus, gamma, beta, dY, m, n, eps, stream);
        const float lm = ForwardLossDotDY(x_minus, gamma, beta, dY, m, n, eps, stream);
        const float g_num = (lp - lm) / (2.0f * fd_eps);
        const float tol = abs_tol + rel_tol * std::fabs(g_num);
        EXPECT_NEAR(dX[i], g_num, tol) << ReproTag("fd_dx", seed, m, n) << " idx=" << i;
    }

    for (size_t i = 0; i < gamma.size(); ++i) {
        std::vector<float> g_plus = gamma;
        std::vector<float> g_minus = gamma;
        g_plus[i] += fd_eps;
        g_minus[i] -= fd_eps;

        const float lp = ForwardLossDotDY(x, g_plus, beta, dY, m, n, eps, stream);
        const float lm = ForwardLossDotDY(x, g_minus, beta, dY, m, n, eps, stream);
        const float g_num = (lp - lm) / (2.0f * fd_eps);
        const float tol = abs_tol + rel_tol * std::fabs(g_num);
        EXPECT_NEAR(dgamma[i], g_num, tol) << ReproTag("fd_dgamma", seed, m, n) << " idx=" << i;
    }

    for (size_t i = 0; i < beta.size(); ++i) {
        std::vector<float> b_plus = beta;
        std::vector<float> b_minus = beta;
        b_plus[i] += fd_eps;
        b_minus[i] -= fd_eps;

        const float lp = ForwardLossDotDY(x, gamma, b_plus, dY, m, n, eps, stream);
        const float lm = ForwardLossDotDY(x, gamma, b_minus, dY, m, n, eps, stream);
        const float g_num = (lp - lm) / (2.0f * fd_eps);
        const float tol = abs_tol + rel_tol * std::fabs(g_num);
        EXPECT_NEAR(dbeta[i], g_num, tol) << ReproTag("fd_dbeta", seed, m, n) << " idx=" << i;
    }
}

TEST(LayerNormBackward, InvariantZeroDYGivesZeroGrads) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 19;
    constexpr int n = 23;
    constexpr float eps = 1e-5f;
    constexpr float zero_abs_tol = 1e-7f;
    const uint32_t seed = fa_test::MixSeed(fa_test::kLayerNormSeedBase, m, n, 410, 0);

    const std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -2.0f, 2.0f, seed);
    const std::vector<float> gamma = fa_test::SampleUniformVector(static_cast<size_t>(n), -2.0f, 2.0f, seed + 1u);
    const std::vector<float> beta = fa_test::SampleUniformVector(static_cast<size_t>(n), -2.0f, 2.0f, seed + 2u);
    const std::vector<float> dY(static_cast<size_t>(m) * n, 0.0f);

    Stream stream;
    Tensor x_d = MakeCpuTensor2D(m, n, x).clone(Device::CUDA, stream);
    Tensor gamma_d = MakeCpuTensor1D(n, gamma).clone(Device::CUDA, stream);
    Tensor beta_d = MakeCpuTensor1D(n, beta).clone(Device::CUDA, stream);
    Tensor dY_d = MakeCpuTensor2D(m, n, dY).clone(Device::CUDA, stream);

    LayerNormResults out = layernorm_forward(x_d, gamma_d, beta_d, eps, &stream);
    LayerNormGrads grads = layernorm_backward(dY_d, out.ctx, true, true, true, &stream);
    stream.synchronize();

    ExpectVectorNear(grads.dX->clone(Device::CPU).to_vector<float>(),
                     std::vector<float>(static_cast<size_t>(m) * n, 0.0f),
                     zero_abs_tol,
                     0.0f,
                     ReproTag("zero_dy_dx", seed, m, n));
    ExpectVectorNear(grads.dgamma->clone(Device::CPU).to_vector<float>(),
                     std::vector<float>(static_cast<size_t>(n), 0.0f),
                     zero_abs_tol,
                     0.0f,
                     ReproTag("zero_dy_dgamma", seed, m, n));
    ExpectVectorNear(grads.dbeta->clone(Device::CPU).to_vector<float>(),
                     std::vector<float>(static_cast<size_t>(n), 0.0f),
                     zero_abs_tol,
                     0.0f,
                     ReproTag("zero_dy_dbeta", seed, m, n));
}

TEST(LayerNormBackward, InvariantLinearityInDYForFixedInputs) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 11;
    constexpr int n = 17;
    constexpr float eps = 1e-5f;
    constexpr float a = 1.7f;
    constexpr float b = -0.4f;
    constexpr float abs_tol = 5e-4f;
    constexpr float rel_tol = 5e-4f;
    const uint32_t seed = fa_test::MixSeed(fa_test::kLayerNormSeedBase, m, n, 420, 0);

    const std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -3.0f, 3.0f, seed);
    const std::vector<float> gamma = fa_test::SampleUniformVector(static_cast<size_t>(n), -3.0f, 3.0f, seed + 1u);
    const std::vector<float> beta = fa_test::SampleUniformVector(static_cast<size_t>(n), -3.0f, 3.0f, seed + 2u);
    const std::vector<float> dY1 = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 3u);
    const std::vector<float> dY2 = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 4u);

    std::vector<float> dYc(dY1.size());
    for (size_t i = 0; i < dY1.size(); ++i) {
        dYc[i] = a * dY1[i] + b * dY2[i];
    }

    Stream stream;
    Tensor x_d = MakeCpuTensor2D(m, n, x).clone(Device::CUDA, stream);
    Tensor gamma_d = MakeCpuTensor1D(n, gamma).clone(Device::CUDA, stream);
    Tensor beta_d = MakeCpuTensor1D(n, beta).clone(Device::CUDA, stream);
    Tensor dY1_d = MakeCpuTensor2D(m, n, dY1).clone(Device::CUDA, stream);
    Tensor dY2_d = MakeCpuTensor2D(m, n, dY2).clone(Device::CUDA, stream);
    Tensor dYc_d = MakeCpuTensor2D(m, n, dYc).clone(Device::CUDA, stream);

    LayerNormResults out = layernorm_forward(x_d, gamma_d, beta_d, eps, &stream);
    LayerNormGrads g1 = layernorm_backward(dY1_d, out.ctx, true, true, true, &stream);
    LayerNormGrads g2 = layernorm_backward(dY2_d, out.ctx, true, true, true, &stream);
    LayerNormGrads gc = layernorm_backward(dYc_d, out.ctx, true, true, true, &stream);
    stream.synchronize();

    const std::vector<float> g1dx = g1.dX->clone(Device::CPU).to_vector<float>();
    const std::vector<float> g2dx = g2.dX->clone(Device::CPU).to_vector<float>();
    const std::vector<float> gcdx = gc.dX->clone(Device::CPU).to_vector<float>();
    const std::vector<float> g1dg = g1.dgamma->clone(Device::CPU).to_vector<float>();
    const std::vector<float> g2dg = g2.dgamma->clone(Device::CPU).to_vector<float>();
    const std::vector<float> gcdg = gc.dgamma->clone(Device::CPU).to_vector<float>();
    const std::vector<float> g1db = g1.dbeta->clone(Device::CPU).to_vector<float>();
    const std::vector<float> g2db = g2.dbeta->clone(Device::CPU).to_vector<float>();
    const std::vector<float> gcdb = gc.dbeta->clone(Device::CPU).to_vector<float>();

    std::vector<float> expected_dx(g1dx.size());
    std::vector<float> expected_dg(g1dg.size());
    std::vector<float> expected_db(g1db.size());
    for (size_t i = 0; i < expected_dx.size(); ++i) {
        expected_dx[i] = a * g1dx[i] + b * g2dx[i];
    }
    for (size_t i = 0; i < expected_dg.size(); ++i) {
        expected_dg[i] = a * g1dg[i] + b * g2dg[i];
        expected_db[i] = a * g1db[i] + b * g2db[i];
    }

    ExpectVectorNear(gcdx, expected_dx, abs_tol, rel_tol, ReproTag("lin_dy_dx", seed, m, n));
    ExpectVectorNear(gcdg, expected_dg, abs_tol, rel_tol, ReproTag("lin_dy_dgamma", seed, m, n));
    ExpectVectorNear(gcdb, expected_db, abs_tol, rel_tol, ReproTag("lin_dy_dbeta", seed, m, n));
}

TEST(LayerNormForwardBackward, CtxStoresPointersAndCachedStats) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 30;
    constexpr int n = 40;
    constexpr float eps = 1e-5f;
    const uint32_t seed = fa_test::MixSeed(fa_test::kLayerNormSeedBase, m, n, 500, 0);

    const std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed);
    const std::vector<float> gamma = fa_test::SampleUniformVector(static_cast<size_t>(n), -1.0f, 1.0f, seed + 1u);
    const std::vector<float> beta = fa_test::SampleUniformVector(static_cast<size_t>(n), -1.0f, 1.0f, seed + 2u);

    Stream stream;
    Tensor x_d = MakeCpuTensor2D(m, n, x).clone(Device::CUDA, stream);
    Tensor gamma_d = MakeCpuTensor1D(n, gamma).clone(Device::CUDA, stream);
    Tensor beta_d = MakeCpuTensor1D(n, beta).clone(Device::CUDA, stream);

    LayerNormResults out = layernorm_forward(x_d, gamma_d, beta_d, eps, &stream);

    ASSERT_EQ(out.ctx.X, &x_d);
    ASSERT_EQ(out.ctx.gamma, &gamma_d);
    ASSERT_EQ(out.ctx.m, m);
    ASSERT_EQ(out.ctx.n, n);
    ASSERT_FLOAT_EQ(out.ctx.eps, eps);
    ASSERT_EQ(out.ctx.mean.shape_.size(), 1u);
    ASSERT_EQ(out.ctx.rstd.shape_.size(), 1u);
    ASSERT_EQ(out.ctx.mean.shape_[0], m);
    ASSERT_EQ(out.ctx.rstd.shape_[0], m);
}

TEST(LayerNormForwardBackward, CtxPointerTracksMutatedGamma) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 13;
    constexpr int n = 11;
    constexpr float eps = 1e-5f;
    constexpr float abs_tol = 5e-4f;
    constexpr float rel_tol = 5e-4f;
    const uint32_t seed = fa_test::MixSeed(fa_test::kLayerNormSeedBase, m, n, 510, 0);

    const std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed);
    const std::vector<float> gamma = fa_test::SampleUniformVector(static_cast<size_t>(n), -1.0f, 1.0f, seed + 1u);
    const std::vector<float> beta = fa_test::SampleUniformVector(static_cast<size_t>(n), -1.0f, 1.0f, seed + 2u);
    const std::vector<float> dY = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 3u);

    Stream stream;
    Tensor x_d = MakeCpuTensor2D(m, n, x).clone(Device::CUDA, stream);
    Tensor gamma_d = MakeCpuTensor1D(n, gamma).clone(Device::CUDA, stream);
    Tensor beta_d = MakeCpuTensor1D(n, beta).clone(Device::CUDA, stream);
    Tensor dY_d = MakeCpuTensor2D(m, n, dY).clone(Device::CUDA, stream);

    LayerNormResults out = layernorm_forward(x_d, gamma_d, beta_d, eps, &stream);
    LayerNormGrads g_before = layernorm_backward(dY_d, out.ctx, true, false, false, &stream);
    ASSERT_TRUE(g_before.dX.has_value());

    ApplyAffineInplaceF32(gamma_d, 0.5f, 0.125f, stream);

    LayerNormGrads g_after = layernorm_backward(dY_d, out.ctx, true, false, false, &stream);
    ASSERT_TRUE(g_after.dX.has_value());

    const std::vector<float> mean = out.ctx.mean.clone(Device::CPU).to_vector<float>();
    const std::vector<float> rstd = out.ctx.rstd.clone(Device::CPU).to_vector<float>();
    const std::vector<float> gamma_after = gamma_d.clone(Device::CPU).to_vector<float>();
    stream.synchronize();

    const fa_test::LayerNormRefBackward ref_after =
        fa_test::reference_layernorm_backward(dY, x, gamma_after, mean, rstd, m, n, true, false, false);

    ExpectVectorNear(g_after.dX->clone(Device::CPU).to_vector<float>(),
                     ref_after.dX,
                     abs_tol,
                     rel_tol,
                     ReproTag("ctx_gamma_mutation", seed, m, n));
}

TEST(LayerNormForwardBackward, TwoStageReuseNoMidTransfer) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 7;
    constexpr int n = 9;
    constexpr float eps = 1e-5f;
    constexpr float abs_tol = 8e-4f;
    constexpr float rel_tol = 8e-4f;
    const uint32_t seed = fa_test::MixSeed(fa_test::kLayerNormSeedBase, m, n, 520, 0);

    const std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed);
    const std::vector<float> gamma = fa_test::SampleUniformVector(static_cast<size_t>(n), -1.0f, 1.0f, seed + 1u);
    const std::vector<float> beta = fa_test::SampleUniformVector(static_cast<size_t>(n), -1.0f, 1.0f, seed + 2u);
    const std::vector<float> dY1 = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 3u);
    const std::vector<float> dY2 = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 4u);

    Stream stream;
    Tensor x_d = MakeCpuTensor2D(m, n, x).clone(Device::CUDA, stream);
    Tensor gamma_d = MakeCpuTensor1D(n, gamma).clone(Device::CUDA, stream);
    Tensor beta_d = MakeCpuTensor1D(n, beta).clone(Device::CUDA, stream);
    Tensor dY1_d = MakeCpuTensor2D(m, n, dY1).clone(Device::CUDA, stream);
    Tensor dY2_d = MakeCpuTensor2D(m, n, dY2).clone(Device::CUDA, stream);

    LayerNormResults out1 = layernorm_forward(x_d, gamma_d, beta_d, eps, &stream);
    LayerNormGrads g1 = layernorm_backward(dY1_d, out1.ctx, true, true, true, &stream);

    ApplyAffineInplaceF32(x_d, 0.5f, 0.125f, stream);
    ApplyAffineInplaceF32(gamma_d, -0.75f, 0.25f, stream);
    ApplyAffineInplaceF32(beta_d, 0.9f, -0.05f, stream);

    LayerNormResults out2 = layernorm_forward(x_d, gamma_d, beta_d, eps, &stream);
    LayerNormGrads g2 = layernorm_backward(dY2_d, out2.ctx, true, true, true, &stream);

    stream.synchronize();

    const std::vector<float> g1_dx = g1.dX->clone(Device::CPU).to_vector<float>();
    const std::vector<float> g1_dg = g1.dgamma->clone(Device::CPU).to_vector<float>();
    const std::vector<float> g1_db = g1.dbeta->clone(Device::CPU).to_vector<float>();
    const std::vector<float> g2_dx = g2.dX->clone(Device::CPU).to_vector<float>();
    const std::vector<float> g2_dg = g2.dgamma->clone(Device::CPU).to_vector<float>();
    const std::vector<float> g2_db = g2.dbeta->clone(Device::CPU).to_vector<float>();

    const std::vector<float> mean1 = out1.ctx.mean.clone(Device::CPU).to_vector<float>();
    const std::vector<float> rstd1 = out1.ctx.rstd.clone(Device::CPU).to_vector<float>();
    const std::vector<float> mean2 = out2.ctx.mean.clone(Device::CPU).to_vector<float>();
    const std::vector<float> rstd2 = out2.ctx.rstd.clone(Device::CPU).to_vector<float>();

    const fa_test::LayerNormRefBackward ref1 =
        fa_test::reference_layernorm_backward(dY1, x, gamma, mean1, rstd1, m, n, true, true, true);

    std::vector<float> x2 = x;
    std::vector<float> gamma2 = gamma;
    std::vector<float> beta2 = beta;
    for (float& v : x2) v = v * 0.5f + 0.125f;
    for (float& v : gamma2) v = v * -0.75f + 0.25f;
    for (float& v : beta2) v = v * 0.9f - 0.05f;

    const fa_test::LayerNormRefBackward ref2 =
        fa_test::reference_layernorm_backward(dY2, x2, gamma2, mean2, rstd2, m, n, true, true, true);

    ExpectVectorNear(g1_dx, ref1.dX, abs_tol, rel_tol, ReproTag("two_stage_s1_dx", seed, m, n));
    ExpectVectorNear(g1_dg, ref1.dgamma, abs_tol, rel_tol, ReproTag("two_stage_s1_dgamma", seed, m, n));
    ExpectVectorNear(g1_db, ref1.dbeta, abs_tol, rel_tol, ReproTag("two_stage_s1_dbeta", seed, m, n));

    ExpectVectorNear(g2_dx, ref2.dX, abs_tol, rel_tol, ReproTag("two_stage_s2_dx", seed, m, n));
    ExpectVectorNear(g2_dg, ref2.dgamma, abs_tol, rel_tol, ReproTag("two_stage_s2_dgamma", seed, m, n));
    ExpectVectorNear(g2_db, ref2.dbeta, abs_tol, rel_tol, ReproTag("two_stage_s2_dbeta", seed, m, n));
}

TEST(LayerNormForwardBackward, CtxIsolationAcrossMultipleForwardsNoMidTransfer) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m1 = 5, n1 = 7;
    constexpr int m2 = 6, n2 = 4;
    constexpr float eps = 1e-5f;
    constexpr float abs_tol = 8e-4f;
    constexpr float rel_tol = 8e-4f;
    const uint32_t seed1 = fa_test::MixSeed(fa_test::kLayerNormSeedBase, m1, n1, 530, 0);
    const uint32_t seed2 = fa_test::MixSeed(fa_test::kLayerNormSeedBase, m2, n2, 531, 0);

    const std::vector<float> x1 = fa_test::SampleUniformVector(static_cast<size_t>(m1) * n1, -1.0f, 1.0f, seed1);
    const std::vector<float> g1 = fa_test::SampleUniformVector(static_cast<size_t>(n1), -1.0f, 1.0f, seed1 + 1u);
    const std::vector<float> b1 = fa_test::SampleUniformVector(static_cast<size_t>(n1), -1.0f, 1.0f, seed1 + 2u);
    const std::vector<float> dY1 = fa_test::SampleUniformVector(static_cast<size_t>(m1) * n1, -1.0f, 1.0f, seed1 + 3u);

    const std::vector<float> x2 = fa_test::SampleUniformVector(static_cast<size_t>(m2) * n2, -1.0f, 1.0f, seed2);
    const std::vector<float> g2 = fa_test::SampleUniformVector(static_cast<size_t>(n2), -1.0f, 1.0f, seed2 + 1u);
    const std::vector<float> b2 = fa_test::SampleUniformVector(static_cast<size_t>(n2), -1.0f, 1.0f, seed2 + 2u);
    const std::vector<float> dY2 = fa_test::SampleUniformVector(static_cast<size_t>(m2) * n2, -1.0f, 1.0f, seed2 + 3u);

    Stream stream;

    Tensor x1_d = MakeCpuTensor2D(m1, n1, x1).clone(Device::CUDA, stream);
    Tensor g1_d = MakeCpuTensor1D(n1, g1).clone(Device::CUDA, stream);
    Tensor b1_d = MakeCpuTensor1D(n1, b1).clone(Device::CUDA, stream);
    Tensor dY1_d = MakeCpuTensor2D(m1, n1, dY1).clone(Device::CUDA, stream);

    Tensor x2_d = MakeCpuTensor2D(m2, n2, x2).clone(Device::CUDA, stream);
    Tensor g2_d = MakeCpuTensor1D(n2, g2).clone(Device::CUDA, stream);
    Tensor b2_d = MakeCpuTensor1D(n2, b2).clone(Device::CUDA, stream);
    Tensor dY2_d = MakeCpuTensor2D(m2, n2, dY2).clone(Device::CUDA, stream);

    LayerNormResults out1 = layernorm_forward(x1_d, g1_d, b1_d, eps, &stream);
    LayerNormResults out2 = layernorm_forward(x2_d, g2_d, b2_d, eps, &stream);

    LayerNormGrads grads2 = layernorm_backward(dY2_d, out2.ctx, true, true, true, &stream);
    LayerNormGrads grads1 = layernorm_backward(dY1_d, out1.ctx, true, true, true, &stream);

    stream.synchronize();

    const fa_test::LayerNormRefBackward ref2 =
        fa_test::reference_layernorm_backward(dY2, x2, g2,
                                              out2.ctx.mean.clone(Device::CPU).to_vector<float>(),
                                              out2.ctx.rstd.clone(Device::CPU).to_vector<float>(),
                                              m2, n2, true, true, true);
    const fa_test::LayerNormRefBackward ref1 =
        fa_test::reference_layernorm_backward(dY1, x1, g1,
                                              out1.ctx.mean.clone(Device::CPU).to_vector<float>(),
                                              out1.ctx.rstd.clone(Device::CPU).to_vector<float>(),
                                              m1, n1, true, true, true);

    ExpectVectorNear(grads2.dX->clone(Device::CPU).to_vector<float>(), ref2.dX, abs_tol, rel_tol,
                     ReproTag("ctx_iso_2_dx", seed2, m2, n2));
    ExpectVectorNear(grads2.dgamma->clone(Device::CPU).to_vector<float>(), ref2.dgamma, abs_tol, rel_tol,
                     ReproTag("ctx_iso_2_dgamma", seed2, m2, n2));
    ExpectVectorNear(grads2.dbeta->clone(Device::CPU).to_vector<float>(), ref2.dbeta, abs_tol, rel_tol,
                     ReproTag("ctx_iso_2_dbeta", seed2, m2, n2));

    ExpectVectorNear(grads1.dX->clone(Device::CPU).to_vector<float>(), ref1.dX, abs_tol, rel_tol,
                     ReproTag("ctx_iso_1_dx", seed1, m1, n1));
    ExpectVectorNear(grads1.dgamma->clone(Device::CPU).to_vector<float>(), ref1.dgamma, abs_tol, rel_tol,
                     ReproTag("ctx_iso_1_dgamma", seed1, m1, n1));
    ExpectVectorNear(grads1.dbeta->clone(Device::CPU).to_vector<float>(), ref1.dbeta, abs_tol, rel_tol,
                     ReproTag("ctx_iso_1_dbeta", seed1, m1, n1));
}

TEST(LayerNormForwardBackward, SweepAllCasesNoMidTransfer) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr float eps = 1e-5f;
    const std::vector<fa_test::LayerNormCase> cases = fa_test::BuildForwardBackwardCases();
    std::vector<std::string> failures;
    failures.reserve(64);

    for (const auto& c : cases) {
        for (int iter = 0; iter < c.iters; ++iter) {
            const uint32_t seed = fa_test::MixSeed(fa_test::kLayerNormSeedBase, c.m, c.n, 600 + iter, 0);
            const std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(c.m) * c.n, c.lo, c.hi, seed);
            const std::vector<float> gamma = fa_test::SampleUniformVector(static_cast<size_t>(c.n), c.lo, c.hi, seed + 1u);
            const std::vector<float> beta = fa_test::SampleUniformVector(static_cast<size_t>(c.n), c.lo, c.hi, seed + 2u);
            const std::vector<float> dY1 = fa_test::SampleUniformVector(static_cast<size_t>(c.m) * c.n, c.lo, c.hi, seed + 3u);
            const std::vector<float> dY2 = fa_test::SampleUniformVector(static_cast<size_t>(c.m) * c.n, c.lo, c.hi, seed + 4u);

            Stream stream;
            Tensor x_d = MakeCpuTensor2D(c.m, c.n, x).clone(Device::CUDA, stream);
            Tensor gamma_d = MakeCpuTensor1D(c.n, gamma).clone(Device::CUDA, stream);
            Tensor beta_d = MakeCpuTensor1D(c.n, beta).clone(Device::CUDA, stream);
            Tensor dY1_d = MakeCpuTensor2D(c.m, c.n, dY1).clone(Device::CUDA, stream);
            Tensor dY2_d = MakeCpuTensor2D(c.m, c.n, dY2).clone(Device::CUDA, stream);

            LayerNormResults out1 = layernorm_forward(x_d, gamma_d, beta_d, eps, &stream);
            LayerNormGrads g1 = layernorm_backward(dY1_d, out1.ctx, true, true, true, &stream);

            ApplyAffineInplaceF32(x_d, 0.5f, 0.125f, stream);
            ApplyAffineInplaceF32(gamma_d, -0.75f, 0.25f, stream);
            ApplyAffineInplaceF32(beta_d, 0.9f, -0.05f, stream);

            LayerNormResults out2 = layernorm_forward(x_d, gamma_d, beta_d, eps, &stream);
            LayerNormGrads g2 = layernorm_backward(dY2_d, out2.ctx, true, true, true, &stream);

            stream.synchronize();

            const std::vector<float> g1dx = g1.dX->clone(Device::CPU).to_vector<float>();
            const std::vector<float> g1dg = g1.dgamma->clone(Device::CPU).to_vector<float>();
            const std::vector<float> g1db = g1.dbeta->clone(Device::CPU).to_vector<float>();
            const std::vector<float> g2dx = g2.dX->clone(Device::CPU).to_vector<float>();
            const std::vector<float> g2dg = g2.dgamma->clone(Device::CPU).to_vector<float>();
            const std::vector<float> g2db = g2.dbeta->clone(Device::CPU).to_vector<float>();

            const fa_test::LayerNormRefBackward ref1 =
                fa_test::reference_layernorm_backward(dY1, x, gamma,
                                                      out1.ctx.mean.clone(Device::CPU).to_vector<float>(),
                                                      out1.ctx.rstd.clone(Device::CPU).to_vector<float>(),
                                                      c.m, c.n, true, true, true);

            std::vector<float> x2 = x;
            std::vector<float> gamma2 = gamma;
            std::vector<float> beta2 = beta;
            for (float& v : x2) v = v * 0.5f + 0.125f;
            for (float& v : gamma2) v = v * -0.75f + 0.25f;
            for (float& v : beta2) v = v * 0.9f - 0.05f;

            const fa_test::LayerNormRefBackward ref2 =
                fa_test::reference_layernorm_backward(dY2, x2, gamma2,
                                                      out2.ctx.mean.clone(Device::CPU).to_vector<float>(),
                                                      out2.ctx.rstd.clone(Device::CPU).to_vector<float>(),
                                                      c.m, c.n, true, true, true);

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

            const int s1_dx_fail = first_fail_idx(g1dx, ref1.dX);
            const int s1_dg_fail = first_fail_idx(g1dg, ref1.dgamma);
            const int s1_db_fail = first_fail_idx(g1db, ref1.dbeta);
            const int s2_dx_fail = first_fail_idx(g2dx, ref2.dX);
            const int s2_dg_fail = first_fail_idx(g2dg, ref2.dgamma);
            const int s2_db_fail = first_fail_idx(g2db, ref2.dbeta);

            if (s1_dx_fail != -1 || s1_dg_fail != -1 || s1_db_fail != -1 ||
                s2_dx_fail != -1 || s2_dg_fail != -1 || s2_db_fail != -1) {
                std::ostringstream one;
                one << "case=" << c.name
                    << " dist=" << fa_test::DistName(c.dist)
                    << " iter=" << iter
                    << " seed=" << seed
                    << " m=" << c.m << " n=" << c.n
                    << " s1_dx_fail_idx=" << s1_dx_fail
                    << " s1_dgamma_fail_idx=" << s1_dg_fail
                    << " s1_dbeta_fail_idx=" << s1_db_fail
                    << " s2_dx_fail_idx=" << s2_dx_fail
                    << " s2_dgamma_fail_idx=" << s2_dg_fail
                    << " s2_dbeta_fail_idx=" << s2_db_fail;
                failures.push_back(one.str());
            }
        }
    }

    if (!failures.empty()) {
        std::ostringstream all;
        all << "LayerNormForwardBackward sweep failed in " << failures.size() << " case(s):\n";
        for (const std::string& f : failures) {
            all << "  " << f << "\n";
        }
        ADD_FAILURE() << all.str();
    }
}

}  // namespace
