#include "general.h"
#include "test_softmax.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
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

std::vector<float> RunForwardToHost(const Tensor& x_h, Stream& stream) {
    Tensor x_d = x_h.clone(Device::CUDA, stream);
    Tensor y_d = softmax_forward(x_d, &stream);
    Tensor y_h = y_d.clone(Device::CPU);
    stream.synchronize();
    return y_h.to_vector<float>();
}

float Dot(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("test_softmax.cu: Dot inputs must have the same size.");
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

float ReferenceMeanCrossEntropy(const std::vector<float>& probs,
                                const std::vector<int32_t>& labels,
                                int m,
                                int n) {
    if (static_cast<int>(labels.size()) != m) {
        throw std::invalid_argument("test_softmax.cu: labels size mismatch in CE reference.");
    }
    double sum = 0.0;
    for (int row = 0; row < m; ++row) {
        const int32_t label = labels[row];
        if (label < 0 || label >= n) {
            throw std::invalid_argument("test_softmax.cu: label out of range in CE reference.");
        }
        const size_t idx = static_cast<size_t>(row) * n + static_cast<size_t>(label);
        const double p = std::max(static_cast<double>(probs[idx]), 1e-12);
        sum += -std::log(p);
    }
    return static_cast<float>(sum / static_cast<double>(m));
}

std::vector<float> ReferenceSoftmaxCrossEntropyGrad(const std::vector<float>& probs,
                                                    const std::vector<int32_t>& labels,
                                                    int m,
                                                    int n) {
    if (static_cast<int>(labels.size()) != m) {
        throw std::invalid_argument("test_softmax.cu: labels size mismatch in CE grad reference.");
    }
    std::vector<float> dX = probs;
    const float inv_m = 1.0f / static_cast<float>(m);
    for (int row = 0; row < m; ++row) {
        const int32_t label = labels[row];
        if (label < 0 || label >= n) {
            throw std::invalid_argument("test_softmax.cu: label out of range in CE grad reference.");
        }
        const size_t idx = static_cast<size_t>(row) * n + static_cast<size_t>(label);
        dX[idx] -= 1.0f;
    }
    for (float& v : dX) {
        v *= inv_m;
    }
    return dX;
}

TEST(SoftmaxForward, RejectsNullStream) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream stream;
    Tensor x({4, 7}, DType::F32, Device::CUDA, stream);
    EXPECT_THROW((void)softmax_forward(x, nullptr), std::invalid_argument);
}

TEST(SoftmaxForward, RejectsNonDefaultStream) {
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

    EXPECT_NO_THROW((void)softmax_forward(x, &non_default_stream));

    CUDA_CHECK(cudaStreamDestroy(raw_non_default));
}

TEST(SoftmaxForward, RejectsNonF32) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream stream;
    Tensor x({4, 7}, DType::F16, Device::CUDA, stream);
    EXPECT_THROW((void)softmax_forward(x, &stream), std::invalid_argument);
}

TEST(SoftmaxForward, RejectsNon2DInput) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream stream;
    Tensor x({4, 7, 2}, DType::F32, Device::CUDA, stream);
    EXPECT_THROW((void)softmax_forward(x, &stream), std::invalid_argument);
}

TEST(SoftmaxForward, TensorRejectsNonPositiveDimsBeforeCall) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream stream;
    EXPECT_THROW((void)Tensor({0, 7}, DType::F32, Device::CUDA, stream), std::invalid_argument);
    EXPECT_THROW((void)Tensor({4, 0}, DType::F32, Device::CUDA, stream), std::invalid_argument);
}

TEST(SoftmaxBackward, RejectsNullStream) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream stream;
    Tensor y({4, 7}, DType::F32, Device::CUDA, stream);
    Tensor dY({4, 7}, DType::F32, Device::CUDA, stream);

    EXPECT_THROW((void)softmax_backward(dY, y, nullptr), std::invalid_argument);
}

TEST(SoftmaxBackward, RejectsNonDefaultStream) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream stream;
    Tensor y({4, 7}, DType::F32, Device::CUDA, stream);
    Tensor dY({4, 7}, DType::F32, Device::CUDA, stream);

    cudaStream_t raw_non_default = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&raw_non_default, cudaStreamNonBlocking));
    Stream non_default_stream;
    non_default_stream.s = raw_non_default;
    non_default_stream.owns_ = false;

    EXPECT_NO_THROW((void)softmax_backward(dY, y, &non_default_stream));

    CUDA_CHECK(cudaStreamDestroy(raw_non_default));
}

TEST(SoftmaxBackward, RejectsNonF32DY) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream stream;
    Tensor y({4, 7}, DType::F32, Device::CUDA, stream);
    Tensor dY({4, 7}, DType::F16, Device::CUDA, stream);

    EXPECT_THROW((void)softmax_backward(dY, y, &stream), std::invalid_argument);
}

TEST(SoftmaxBackward, RejectsNonF32Y) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream stream;
    Tensor y({4, 7}, DType::F16, Device::CUDA, stream);
    Tensor dY({4, 7}, DType::F32, Device::CUDA, stream);

    EXPECT_THROW((void)softmax_backward(dY, y, &stream), std::invalid_argument);
}

TEST(SoftmaxBackward, RejectsDeviceMismatchYDY) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream stream;
    Tensor y_cuda({4, 7}, DType::F32, Device::CUDA, stream);
    Tensor dY_cpu({4, 7}, DType::F32, Device::CPU);

    EXPECT_THROW((void)softmax_backward(dY_cpu, y_cuda, &stream), std::invalid_argument);
}

TEST(SoftmaxBackward, RejectsShapeMismatch) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream stream;
    Tensor y({4, 7}, DType::F32, Device::CUDA, stream);
    Tensor dY({7, 4}, DType::F32, Device::CUDA, stream);

    EXPECT_THROW((void)softmax_backward(dY, y, &stream), std::invalid_argument);
}

TEST(SoftmaxBackward, RejectsNon2DY) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream stream;
    Tensor y({4, 7, 2}, DType::F32, Device::CUDA, stream);
    Tensor dY({4, 7, 2}, DType::F32, Device::CUDA, stream);

    EXPECT_THROW((void)softmax_backward(dY, y, &stream), std::invalid_argument);
}

TEST(SoftmaxBackward, TensorRejectsNonPositiveDimsBeforeCall) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    Stream stream;
    EXPECT_THROW((void)Tensor({0, 7}, DType::F32, Device::CUDA, stream), std::invalid_argument);
    EXPECT_THROW((void)Tensor({4, 0}, DType::F32, Device::CUDA, stream), std::invalid_argument);
}

TEST(SoftmaxForward, MatchesReferenceOddShape) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 13;
    constexpr int n = 37;
    constexpr float abs_tol = 2e-5f;
    constexpr float rel_tol = 2e-5f;

    const uint32_t seed = fa_test::MixSeed(fa_test::kSoftmaxSeedBase, m, n, 101, 0);
    const std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed);

    Stream stream;
    Tensor x_d = MakeCpuTensor2D(m, n, x).clone(Device::CUDA, stream);

    Tensor y_d = softmax_forward(x_d, &stream);
    std::vector<float> y = y_d.clone(Device::CPU).to_vector<float>();
    stream.synchronize();

    const std::vector<float> expected = fa_test::reference_softmax_forward(x, m, n);
    ExpectVectorNear(y, expected, abs_tol, rel_tol, ReproTag("forward_odd", seed, m, n));
}

TEST(SoftmaxBackward, MatchesReferenceOddShape) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 13;
    constexpr int n = 37;
    constexpr float abs_tol = 3e-5f;
    constexpr float rel_tol = 3e-5f;

    const uint32_t seed = fa_test::MixSeed(fa_test::kSoftmaxSeedBase, m, n, 201, 0);
    const std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed);
    const std::vector<float> dY = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 1u);

    Stream stream;
    Tensor x_d = MakeCpuTensor2D(m, n, x).clone(Device::CUDA, stream);
    Tensor dY_d = MakeCpuTensor2D(m, n, dY).clone(Device::CUDA, stream);

    Tensor y_d = softmax_forward(x_d, &stream);
    SoftmaxGrads grads = softmax_backward(dY_d, y_d, &stream);

    std::vector<float> y = y_d.clone(Device::CPU).to_vector<float>();
    std::vector<float> dX = grads.dX.clone(Device::CPU).to_vector<float>();
    stream.synchronize();

    const std::vector<float> y_expected = fa_test::reference_softmax_forward(x, m, n);
    const std::vector<float> dX_expected = fa_test::reference_softmax_backward(dY, y_expected, m, n);

    ExpectVectorNear(y, y_expected, abs_tol, rel_tol, ReproTag("backward_y_odd", seed, m, n));
    ExpectVectorNear(dX, dX_expected, abs_tol, rel_tol, ReproTag("backward_dx_odd", seed, m, n));
}

TEST(SoftmaxForward, SweepAllCases) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    const std::vector<fa_test::SoftmaxCase> cases = fa_test::BuildForwardCases();
    std::vector<std::string> failures;

    for (const auto& c : cases) {
        for (int iter = 0; iter < c.iters; ++iter) {
            const uint32_t seed = fa_test::MixSeed(fa_test::kSoftmaxSeedBase, c.m, c.n, iter, 1);
            const std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(c.m) * c.n, c.lo, c.hi, seed);

            Stream stream;
            Tensor x_d = MakeCpuTensor2D(c.m, c.n, x).clone(Device::CUDA, stream);
            Tensor y_d = softmax_forward(x_d, &stream);
            std::vector<float> y = y_d.clone(Device::CPU).to_vector<float>();
            stream.synchronize();

            const std::vector<float> expected = fa_test::reference_softmax_forward(x, c.m, c.n);

            int fail_count = 0;
            float worst_abs_err = 0.0f;
            float worst_tol = 0.0f;
            int worst_idx = -1;
            for (size_t i = 0; i < y.size(); ++i) {
                const float abs_err = std::fabs(y[i] - expected[i]);
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
        all << "SoftmaxForward sweep failed in " << failures.size() << " case(s):\n";
        for (const auto& f : failures) {
            all << "  " << f << "\n";
        }
        ADD_FAILURE() << all.str();
    }
}

TEST(SoftmaxCrossEntropy, ForwardMatchesReferenceOddShape) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 13;
    constexpr int n = 37;
    constexpr float abs_tol = 3e-5f;
    constexpr float rel_tol = 3e-5f;
    const uint32_t seed = fa_test::MixSeed(fa_test::kSoftmaxSeedBase, m, n, 301, 0);

    const std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed);
    std::vector<int32_t> labels(static_cast<size_t>(m), 0);
    for (int i = 0; i < m; ++i) {
        labels[static_cast<size_t>(i)] = static_cast<int32_t>((seed + static_cast<uint32_t>(i * 17)) % static_cast<uint32_t>(n));
    }

    Stream stream;
    Tensor x_d = MakeCpuTensor2D(m, n, x).clone(Device::CUDA, stream);
    Tensor labels_h({m}, DType::I32, Device::CPU);
    labels_h.copy_from(labels);
    Tensor labels_d = labels_h.clone(Device::CUDA, stream);

    SoftmaxCrossEntropyResults out = softmax_cross_entropy_forward(x_d, labels_d, &stream);
    const float loss = out.loss.clone(Device::CPU).to_vector<float>()[0];
    stream.synchronize();

    const std::vector<float> probs_ref = fa_test::reference_softmax_forward(x, m, n);
    const float loss_ref = ReferenceMeanCrossEntropy(probs_ref, labels, m, n);
    const float tol = abs_tol + rel_tol * std::fabs(loss_ref);
    EXPECT_NEAR(loss, loss_ref, tol);
}

TEST(SoftmaxCrossEntropy, BackwardMatchesReferenceOddShape) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 13;
    constexpr int n = 37;
    constexpr float abs_tol = 3e-5f;
    constexpr float rel_tol = 3e-5f;
    const uint32_t seed = fa_test::MixSeed(fa_test::kSoftmaxSeedBase, m, n, 401, 0);

    const std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed);
    std::vector<int32_t> labels(static_cast<size_t>(m), 0);
    for (int i = 0; i < m; ++i) {
        labels[static_cast<size_t>(i)] = static_cast<int32_t>((seed + static_cast<uint32_t>(i * 19)) % static_cast<uint32_t>(n));
    }

    Stream stream;
    Tensor x_d = MakeCpuTensor2D(m, n, x).clone(Device::CUDA, stream);
    Tensor labels_h({m}, DType::I32, Device::CPU);
    labels_h.copy_from(labels);
    Tensor labels_d = labels_h.clone(Device::CUDA, stream);

    SoftmaxCrossEntropyResults out = softmax_cross_entropy_forward(x_d, labels_d, &stream);
    SoftmaxCrossEntropyGrads grads = softmax_cross_entropy_backward(out.ctx, &stream);
    std::vector<float> dX = grads.dX.clone(Device::CPU).to_vector<float>();
    stream.synchronize();

    const std::vector<float> probs_ref = fa_test::reference_softmax_forward(x, m, n);
    const std::vector<float> dX_ref = ReferenceSoftmaxCrossEntropyGrad(probs_ref, labels, m, n);
    ExpectVectorNear(dX, dX_ref, abs_tol, rel_tol, ReproTag("softmax_ce_backward_odd", seed, m, n));
}

TEST(SoftmaxBackward, SweepAllCases) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    const std::vector<fa_test::SoftmaxCase> cases = fa_test::BuildBackwardCases();
    std::vector<std::string> failures;

    for (const auto& c : cases) {
        for (int iter = 0; iter < c.iters; ++iter) {
            const uint32_t seed = fa_test::MixSeed(fa_test::kSoftmaxSeedBase, c.m, c.n, iter, 2);
            const std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(c.m) * c.n, c.lo, c.hi, seed);
            const std::vector<float> dY = fa_test::SampleUniformVector(static_cast<size_t>(c.m) * c.n, c.lo, c.hi, seed + 1u);

            Stream stream;
            Tensor x_d = MakeCpuTensor2D(c.m, c.n, x).clone(Device::CUDA, stream);
            Tensor dY_d = MakeCpuTensor2D(c.m, c.n, dY).clone(Device::CUDA, stream);

            Tensor y_d = softmax_forward(x_d, &stream);
            SoftmaxGrads grads = softmax_backward(dY_d, y_d, &stream);

            std::vector<float> y = y_d.clone(Device::CPU).to_vector<float>();
            std::vector<float> dX = grads.dX.clone(Device::CPU).to_vector<float>();
            stream.synchronize();

            const std::vector<float> y_expected = fa_test::reference_softmax_forward(x, c.m, c.n);
            const std::vector<float> dX_expected = fa_test::reference_softmax_backward(dY, y_expected, c.m, c.n);

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

            const int y_fail = first_fail_idx(y, y_expected);
            const int dx_fail = first_fail_idx(dX, dX_expected);
            if (y_fail != -1 || dx_fail != -1) {
                std::ostringstream one;
                one << "case=" << c.name
                    << " dist=" << fa_test::DistName(c.dist)
                    << " iter=" << iter
                    << " seed=" << seed
                    << " m=" << c.m << " n=" << c.n
                    << " y_fail_idx=" << y_fail
                    << " dx_fail_idx=" << dx_fail;
                failures.push_back(one.str());
            }
        }
    }

    if (!failures.empty()) {
        std::ostringstream all;
        all << "SoftmaxBackward sweep failed in " << failures.size() << " case(s):\n";
        for (const auto& f : failures) {
            all << "  " << f << "\n";
        }
        ADD_FAILURE() << all.str();
    }
}

TEST(SoftmaxForward, NumericEdgePatternsAndShapes) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    std::vector<fa_test::SoftmaxEdgeCase> cases;
    cases.push_back(fa_test::SoftmaxEdgeCase{"zeros_11", 1, 1, 1e-6f, 1e-6f, {0.0f}});
    cases.push_back(fa_test::SoftmaxEdgeCase{"single_row_extreme", 1, 5, 5e-5f, 5e-5f,
                                              {-80.0f, -1.0f, 0.0f, 1.0f, 80.0f}});
    cases.push_back(fa_test::SoftmaxEdgeCase{"n_is_one", 7, 1, 1e-6f, 1e-6f,
                                              {-3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f}});
    cases.push_back(fa_test::SoftmaxEdgeCase{"constant_rows_34", 3, 4, 2e-6f, 2e-6f,
                                              {2.5f, 2.5f, 2.5f, 2.5f,
                                               -7.0f, -7.0f, -7.0f, -7.0f,
                                               0.0f, 0.0f, 0.0f, 0.0f}});

    for (const auto& c : cases) {
        Stream stream;
        Tensor x_d = MakeCpuTensor2D(c.m, c.n, c.x).clone(Device::CUDA, stream);
        Tensor y_d = softmax_forward(x_d, &stream);

        const std::vector<float> got = y_d.clone(Device::CPU).to_vector<float>();
        stream.synchronize();
        const std::vector<float> expected = fa_test::reference_softmax_forward(c.x, c.m, c.n);

        ExpectVectorNear(got, expected, c.abs_tol, c.rel_tol,
                         std::string("edge=") + c.name + " m=" + std::to_string(c.m) + " n=" + std::to_string(c.n));
    }
}

TEST(SoftmaxForward, InvariantRowSumsToOneAndRange) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 41;
    constexpr int n = 67;
    const uint32_t seed = fa_test::MixSeed(fa_test::kSoftmaxSeedBase, m, n, 401, 0);
    const std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -10.0f, 10.0f, seed);

    Stream stream;
    Tensor x_d = MakeCpuTensor2D(m, n, x).clone(Device::CUDA, stream);
    Tensor y_d = softmax_forward(x_d, &stream);
    std::vector<float> y = y_d.clone(Device::CPU).to_vector<float>();
    stream.synchronize();

    for (int row = 0; row < m; ++row) {
        double row_sum = 0.0;
        for (int col = 0; col < n; ++col) {
            const float v = y[static_cast<size_t>(row) * n + col];
            EXPECT_GE(v, -1e-6f) << ReproTag("range_low", seed, m, n) << " row=" << row << " col=" << col;
            EXPECT_LE(v, 1.0f + 1e-6f) << ReproTag("range_high", seed, m, n) << " row=" << row << " col=" << col;
            row_sum += static_cast<double>(v);
        }
        EXPECT_NEAR(row_sum, 1.0, 5e-5) << ReproTag("rowsum", seed, m, n) << " row=" << row;
    }
}

TEST(SoftmaxForward, InvariantShiftInvariancePerRow) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 19;
    constexpr int n = 23;
    constexpr float abs_tol = 3e-5f;
    constexpr float rel_tol = 3e-5f;

    const uint32_t seed = fa_test::MixSeed(fa_test::kSoftmaxSeedBase, m, n, 402, 0);
    std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -3.0f, 3.0f, seed);
    const std::vector<float> row_shift = fa_test::SampleUniformVector(static_cast<size_t>(m), -100.0f, 100.0f, seed + 1u);

    std::vector<float> x_shifted = x;
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            x_shifted[static_cast<size_t>(row) * n + col] += row_shift[static_cast<size_t>(row)];
        }
    }

    Stream stream;
    const std::vector<float> y = RunForwardToHost(MakeCpuTensor2D(m, n, x), stream);
    const std::vector<float> y_shifted = RunForwardToHost(MakeCpuTensor2D(m, n, x_shifted), stream);

    ExpectVectorNear(y_shifted, y, abs_tol, rel_tol, ReproTag("shift_invariance", seed, m, n));
}

TEST(SoftmaxForward, InvariantDeterministicForSameInput) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 31;
    constexpr int n = 47;
    constexpr float abs_tol = 1e-6f;
    constexpr float rel_tol = 1e-6f;

    const uint32_t seed = fa_test::MixSeed(fa_test::kSoftmaxSeedBase, m, n, 403, 0);
    const std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed);

    Stream stream;
    Tensor x_d = MakeCpuTensor2D(m, n, x).clone(Device::CUDA, stream);
    Tensor y1_d = softmax_forward(x_d, &stream);
    Tensor y2_d = softmax_forward(x_d, &stream);

    const std::vector<float> y1 = y1_d.clone(Device::CPU).to_vector<float>();
    const std::vector<float> y2 = y2_d.clone(Device::CPU).to_vector<float>();
    stream.synchronize();

    ExpectVectorNear(y2, y1, abs_tol, rel_tol, ReproTag("deterministic_forward", seed, m, n));
}

TEST(SoftmaxBackward, FiniteDifferenceGradientCheckSmall) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    struct Shape { int m; int n; };
    const std::vector<Shape> shapes = {{2, 3}, {3, 4}};

    constexpr float eps = 1e-3f;
    constexpr float abs_tol = 8e-3f;
    constexpr float rel_tol = 8e-3f;

    for (const auto& s : shapes) {
        const uint32_t seed = fa_test::MixSeed(fa_test::kSoftmaxSeedBase, s.m, s.n, 501, 0);
        const size_t numel = static_cast<size_t>(s.m) * s.n;

        std::vector<float> x = fa_test::SampleUniformVector(numel, -2.0f, 2.0f, seed);
        const std::vector<float> dY_ref = fa_test::SampleUniformVector(numel, -1.0f, 1.0f, seed + 1u);

        Stream stream;
        Tensor x_d = MakeCpuTensor2D(s.m, s.n, x).clone(Device::CUDA, stream);
        Tensor dY_d = MakeCpuTensor2D(s.m, s.n, dY_ref).clone(Device::CUDA, stream);
        Tensor y_d = softmax_forward(x_d, &stream);
        SoftmaxGrads grads = softmax_backward(dY_d, y_d, &stream);
        const std::vector<float> dX_analytic = grads.dX.clone(Device::CPU).to_vector<float>();
        stream.synchronize();

        std::vector<float> dX_numeric(numel, 0.0f);
        for (size_t i = 0; i < numel; ++i) {
            std::vector<float> x_plus = x;
            std::vector<float> x_minus = x;
            x_plus[i] += eps;
            x_minus[i] -= eps;

            const float l_plus = ForwardLossDotDY(x_plus, dY_ref, s.m, s.n, stream);
            const float l_minus = ForwardLossDotDY(x_minus, dY_ref, s.m, s.n, stream);
            dX_numeric[i] = (l_plus - l_minus) / (2.0f * eps);
        }

        ExpectVectorNear(dX_analytic, dX_numeric, abs_tol, rel_tol,
                         ReproTag("finite_diff", seed, s.m, s.n));
    }
}

TEST(SoftmaxBackward, InvariantRowSumOfDXIsZero) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 29;
    constexpr int n = 61;

    const uint32_t seed = fa_test::MixSeed(fa_test::kSoftmaxSeedBase, m, n, 601, 0);
    const std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -3.0f, 3.0f, seed);
    const std::vector<float> dY = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 1u);

    Stream stream;
    Tensor x_d = MakeCpuTensor2D(m, n, x).clone(Device::CUDA, stream);
    Tensor dY_d = MakeCpuTensor2D(m, n, dY).clone(Device::CUDA, stream);

    Tensor y_d = softmax_forward(x_d, &stream);
    SoftmaxGrads grads = softmax_backward(dY_d, y_d, &stream);
    std::vector<float> dX = grads.dX.clone(Device::CPU).to_vector<float>();
    stream.synchronize();

    for (int row = 0; row < m; ++row) {
        double sum = 0.0;
        for (int col = 0; col < n; ++col) {
            sum += static_cast<double>(dX[static_cast<size_t>(row) * n + col]);
        }
        EXPECT_NEAR(sum, 0.0, 8e-5) << ReproTag("row_dx_sum_zero", seed, m, n) << " row=" << row;
    }
}

TEST(SoftmaxBackward, InvariantZeroDYGivesZeroDX) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 17;
    constexpr int n = 41;

    const uint32_t seed = fa_test::MixSeed(fa_test::kSoftmaxSeedBase, m, n, 602, 0);
    const std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -4.0f, 4.0f, seed);
    const std::vector<float> dY_zero(static_cast<size_t>(m) * n, 0.0f);

    Stream stream;
    Tensor x_d = MakeCpuTensor2D(m, n, x).clone(Device::CUDA, stream);
    Tensor dY_d = MakeCpuTensor2D(m, n, dY_zero).clone(Device::CUDA, stream);

    Tensor y_d = softmax_forward(x_d, &stream);
    SoftmaxGrads grads = softmax_backward(dY_d, y_d, &stream);
    std::vector<float> dX = grads.dX.clone(Device::CPU).to_vector<float>();
    stream.synchronize();

    for (size_t i = 0; i < dX.size(); ++i) {
        EXPECT_NEAR(dX[i], 0.0f, 1e-7f) << ReproTag("zero_dy", seed, m, n) << " idx=" << i;
    }
}

TEST(SoftmaxBackward, InvariantLinearityInDYForFixedY) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 13;
    constexpr int n = 37;
    constexpr float abs_tol = 4e-5f;
    constexpr float rel_tol = 4e-5f;

    const uint32_t seed = fa_test::MixSeed(fa_test::kSoftmaxSeedBase, m, n, 603, 0);
    const std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -2.0f, 2.0f, seed);
    const std::vector<float> dY1 = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 1u);
    const std::vector<float> dY2 = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 2u);

    std::vector<float> dYc(dY1.size(), 0.0f);
    for (size_t i = 0; i < dY1.size(); ++i) {
        dYc[i] = dY1[i] + dY2[i];
    }

    Stream stream;
    Tensor x_d = MakeCpuTensor2D(m, n, x).clone(Device::CUDA, stream);
    Tensor y_d = softmax_forward(x_d, &stream);

    SoftmaxGrads g1 = softmax_backward(MakeCpuTensor2D(m, n, dY1).clone(Device::CUDA, stream), y_d, &stream);
    SoftmaxGrads g2 = softmax_backward(MakeCpuTensor2D(m, n, dY2).clone(Device::CUDA, stream), y_d, &stream);
    SoftmaxGrads gc = softmax_backward(MakeCpuTensor2D(m, n, dYc).clone(Device::CUDA, stream), y_d, &stream);

    const std::vector<float> dX1 = g1.dX.clone(Device::CPU).to_vector<float>();
    const std::vector<float> dX2 = g2.dX.clone(Device::CPU).to_vector<float>();
    const std::vector<float> dXc = gc.dX.clone(Device::CPU).to_vector<float>();
    stream.synchronize();

    std::vector<float> expected(dX1.size(), 0.0f);
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = dX1[i] + dX2[i];
    }

    ExpectVectorNear(dXc, expected, abs_tol, rel_tol, ReproTag("linearity_dy", seed, m, n));
}

TEST(SoftmaxBackward, InvariantDeterministicForSameYAndDY) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 27;
    constexpr int n = 33;
    constexpr float abs_tol = 1e-6f;
    constexpr float rel_tol = 1e-6f;

    const uint32_t seed = fa_test::MixSeed(fa_test::kSoftmaxSeedBase, m, n, 604, 0);
    const std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed);
    const std::vector<float> dY = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 1u);

    Stream stream;
    Tensor x_d = MakeCpuTensor2D(m, n, x).clone(Device::CUDA, stream);
    Tensor dY_d = MakeCpuTensor2D(m, n, dY).clone(Device::CUDA, stream);

    Tensor y_d = softmax_forward(x_d, &stream);
    SoftmaxGrads g1 = softmax_backward(dY_d, y_d, &stream);
    SoftmaxGrads g2 = softmax_backward(dY_d, y_d, &stream);

    const std::vector<float> dX1 = g1.dX.clone(Device::CPU).to_vector<float>();
    const std::vector<float> dX2 = g2.dX.clone(Device::CPU).to_vector<float>();
    stream.synchronize();

    ExpectVectorNear(dX2, dX1, abs_tol, rel_tol, ReproTag("deterministic_backward", seed, m, n));
}

TEST(SoftmaxForwardBackward, TwoStageReuseNoMidTransfer) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 13;
    constexpr int n = 37;
    constexpr float abs_tol = 3e-5f;
    constexpr float rel_tol = 3e-5f;

    const uint32_t seed = fa_test::MixSeed(fa_test::kSoftmaxSeedBase, m, n, 701, 0);
    const std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed);
    const std::vector<float> dY = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 1u);

    Stream stream;
    Tensor x_d = MakeCpuTensor2D(m, n, x).clone(Device::CUDA, stream);
    Tensor dY_d = MakeCpuTensor2D(m, n, dY).clone(Device::CUDA, stream);

    Tensor y_d = softmax_forward(x_d, &stream);
    SoftmaxGrads grads = softmax_backward(dY_d, y_d, &stream);

    const std::vector<float> y = y_d.clone(Device::CPU).to_vector<float>();
    const std::vector<float> dX = grads.dX.clone(Device::CPU).to_vector<float>();
    stream.synchronize();

    const std::vector<float> y_expected = fa_test::reference_softmax_forward(x, m, n);
    const std::vector<float> dX_expected = fa_test::reference_softmax_backward(dY, y_expected, m, n);

    ExpectVectorNear(y, y_expected, abs_tol, rel_tol, ReproTag("fwd_bwd_y", seed, m, n));
    ExpectVectorNear(dX, dX_expected, abs_tol, rel_tol, ReproTag("fwd_bwd_dx", seed, m, n));
}

TEST(SoftmaxForwardBackward, IsolationAcrossMultipleForwardsNoMidTransfer) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA runtime unavailable";
    }

    constexpr int m = 9;
    constexpr int n = 17;
    constexpr float abs_tol = 4e-5f;
    constexpr float rel_tol = 4e-5f;

    const uint32_t seed = fa_test::MixSeed(fa_test::kSoftmaxSeedBase, m, n, 702, 0);
    const std::vector<float> x1 = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed);
    const std::vector<float> x2 = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 1u);
    const std::vector<float> dY1 = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 2u);
    const std::vector<float> dY2 = fa_test::SampleUniformVector(static_cast<size_t>(m) * n, -1.0f, 1.0f, seed + 3u);

    Stream stream;
    Tensor x1_d = MakeCpuTensor2D(m, n, x1).clone(Device::CUDA, stream);
    Tensor x2_d = MakeCpuTensor2D(m, n, x2).clone(Device::CUDA, stream);
    Tensor dY1_d = MakeCpuTensor2D(m, n, dY1).clone(Device::CUDA, stream);
    Tensor dY2_d = MakeCpuTensor2D(m, n, dY2).clone(Device::CUDA, stream);

    Tensor y1_d = softmax_forward(x1_d, &stream);
    Tensor y2_d = softmax_forward(x2_d, &stream);

    SoftmaxGrads g2 = softmax_backward(dY2_d, y2_d, &stream);
    SoftmaxGrads g1 = softmax_backward(dY1_d, y1_d, &stream);

    const std::vector<float> dX1 = g1.dX.clone(Device::CPU).to_vector<float>();
    const std::vector<float> dX2 = g2.dX.clone(Device::CPU).to_vector<float>();
    stream.synchronize();

    const std::vector<float> y1_ref = fa_test::reference_softmax_forward(x1, m, n);
    const std::vector<float> y2_ref = fa_test::reference_softmax_forward(x2, m, n);
    const std::vector<float> dX1_ref = fa_test::reference_softmax_backward(dY1, y1_ref, m, n);
    const std::vector<float> dX2_ref = fa_test::reference_softmax_backward(dY2, y2_ref, m, n);

    ExpectVectorNear(dX1, dX1_ref, abs_tol, rel_tol, ReproTag("isolation_1", seed, m, n));
    ExpectVectorNear(dX2, dX2_ref, abs_tol, rel_tol, ReproTag("isolation_2", seed, m, n));
}

TEST(SoftmaxForwardBackward, SweepAllCasesNoMidTransfer) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    const std::vector<fa_test::SoftmaxCase> cases = fa_test::BuildForwardBackwardCases();
    std::vector<std::string> failures;

    for (const auto& c : cases) {
        for (int iter = 0; iter < c.iters; ++iter) {
            const uint32_t seed = fa_test::MixSeed(fa_test::kSoftmaxSeedBase, c.m, c.n, iter, 3);
            const std::vector<float> x = fa_test::SampleUniformVector(static_cast<size_t>(c.m) * c.n, c.lo, c.hi, seed);
            const std::vector<float> dY = fa_test::SampleUniformVector(static_cast<size_t>(c.m) * c.n, c.lo, c.hi, seed + 1u);

            Stream stream;
            Tensor x_d = MakeCpuTensor2D(c.m, c.n, x).clone(Device::CUDA, stream);
            Tensor dY_d = MakeCpuTensor2D(c.m, c.n, dY).clone(Device::CUDA, stream);

            Tensor y_d = softmax_forward(x_d, &stream);
            SoftmaxGrads grads = softmax_backward(dY_d, y_d, &stream);

            const std::vector<float> y = y_d.clone(Device::CPU).to_vector<float>();
            const std::vector<float> dX = grads.dX.clone(Device::CPU).to_vector<float>();
            stream.synchronize();

            const std::vector<float> y_expected = fa_test::reference_softmax_forward(x, c.m, c.n);
            const std::vector<float> dX_expected = fa_test::reference_softmax_backward(dY, y_expected, c.m, c.n);

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

            const int y_fail = first_fail_idx(y, y_expected);
            const int dx_fail = first_fail_idx(dX, dX_expected);
            if (y_fail != -1 || dx_fail != -1) {
                std::ostringstream one;
                one << "case=" << c.name
                    << " dist=" << fa_test::DistName(c.dist)
                    << " iter=" << iter
                    << " seed=" << seed
                    << " m=" << c.m << " n=" << c.n
                    << " y_fail_idx=" << y_fail
                    << " dx_fail_idx=" << dx_fail;
                failures.push_back(one.str());
            }
        }
    }

    if (!failures.empty()) {
        std::ostringstream all;
        all << "SoftmaxForwardBackward sweep failed in " << failures.size() << " case(s):\n";
        for (const auto& f : failures) {
            all << "  " << f << "\n";
        }
        ADD_FAILURE() << all.str();
    }
}

}  // namespace
