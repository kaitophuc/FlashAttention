#include "cu_stream.h"
#include "general.h"
#include "ops.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace {

int32_t ReferenceCorrectCount(const std::vector<float>& logits,
                              const std::vector<int32_t>& labels,
                              int64_t m,
                              int64_t n) {
    int32_t correct = 0;
    for (int64_t row = 0; row < m; ++row) {
        const int64_t row_offset = row * n;
        float best_val = logits[static_cast<size_t>(row_offset)];
        int32_t best_idx = 0;
        for (int64_t col = 1; col < n; ++col) {
            const float v = logits[static_cast<size_t>(row_offset + col)];
            if (v > best_val) {
                best_val = v;
                best_idx = static_cast<int32_t>(col);
            }
        }
        if (best_idx == labels[static_cast<size_t>(row)]) {
            ++correct;
        }
    }
    return correct;
}

TEST(ClassificationCorrectCount, MatchesReferenceSmallKnownCase) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    Stream stream;
    constexpr int64_t m = 4;
    constexpr int64_t n = 3;
    Tensor logits({m, n}, DType::F32, Device::CUDA, stream);
    Tensor labels({m}, DType::I32, Device::CUDA, stream);

    const std::vector<float> logits_h = {
        2.0f, 1.0f, 0.0f,   // pred 0, label 0 -> correct
        0.1f, 0.7f, 0.2f,   // pred 1, label 1 -> correct
        -1.0f, 0.0f, 1.0f,  // pred 2, label 0 -> incorrect
        0.2f, 0.9f, 0.3f,   // pred 1, label 1 -> correct
    };
    const std::vector<int32_t> labels_h = {0, 1, 0, 1};

    logits.copy_from(logits_h, stream);
    labels.copy_from(labels_h, stream);

    Tensor correct_d = classification_correct_count(logits, labels, &stream);
    const std::vector<int32_t> got = correct_d.to_vector<int32_t>(stream);
    ASSERT_EQ(got.size(), 1u);
    EXPECT_EQ(got[0], 3);
}

TEST(ClassificationCorrectCount, MatchesReferenceLargerAndTieBehavior) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    Stream stream;
    constexpr int64_t m = 6;
    constexpr int64_t n = 5;
    Tensor logits({m, n}, DType::F32, Device::CUDA, stream);
    Tensor labels({m}, DType::I32, Device::CUDA, stream);

    const std::vector<float> logits_h = {
        1.0f, 1.0f, 0.2f, -0.5f, 0.0f,   // tie at 0/1 -> pick 0
        -1.0f, -0.2f, -0.3f, -0.1f, -0.9f,  // pick 3
        2.5f, 2.4f, 2.3f, 2.2f, 2.1f,    // pick 0
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,    // full tie -> pick 0
        -3.0f, 4.0f, 2.0f, 1.0f, 0.0f,   // pick 1
        0.5f, 0.4f, 0.3f, 0.2f, 0.1f,    // pick 0
    };
    const std::vector<int32_t> labels_h = {
        0,  // correct (tie winner first index)
        3,  // correct
        4,  // incorrect
        0,  // correct
        1,  // correct
        2,  // incorrect
    };

    logits.copy_from(logits_h, stream);
    labels.copy_from(labels_h, stream);

    const int32_t expected = ReferenceCorrectCount(logits_h, labels_h, m, n);
    Tensor correct_d = classification_correct_count(logits, labels, &stream);
    const std::vector<int32_t> got = correct_d.to_vector<int32_t>(stream);
    ASSERT_EQ(got.size(), 1u);
    EXPECT_EQ(got[0], expected);
}

TEST(ClassificationCorrectCount, RejectsInvalidInputs) {
    if (!fa_test::cuda_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    Stream stream;
    Tensor logits_f32({2, 3}, DType::F32, Device::CUDA, stream);
    Tensor labels_i32({2}, DType::I32, Device::CUDA, stream);
    Tensor logits_i32({2, 3}, DType::I32, Device::CUDA, stream);
    Tensor labels_f32({2}, DType::F32, Device::CUDA, stream);
    Tensor labels_bad_shape({2, 1}, DType::I32, Device::CUDA, stream);
    Tensor labels_bad_len({3}, DType::I32, Device::CUDA, stream);
    Tensor logits_3d({2, 3, 1}, DType::F32, Device::CUDA, stream);
    Tensor logits_cpu({2, 3}, DType::F32, Device::CPU);
    Tensor labels_cpu({2}, DType::I32, Device::CPU);

    EXPECT_THROW((void)classification_correct_count(logits_f32, labels_i32, nullptr), std::invalid_argument);
    EXPECT_THROW((void)classification_correct_count(logits_i32, labels_i32, &stream), std::invalid_argument);
    EXPECT_THROW((void)classification_correct_count(logits_f32, labels_f32, &stream), std::invalid_argument);
    EXPECT_THROW((void)classification_correct_count(logits_3d, labels_i32, &stream), std::invalid_argument);
    EXPECT_THROW((void)classification_correct_count(logits_f32, labels_bad_shape, &stream), std::invalid_argument);
    EXPECT_THROW((void)classification_correct_count(logits_f32, labels_bad_len, &stream), std::invalid_argument);
    EXPECT_THROW((void)classification_correct_count(logits_cpu, labels_i32, &stream), std::invalid_argument);
    EXPECT_THROW((void)classification_correct_count(logits_f32, labels_cpu, &stream), std::invalid_argument);

    cudaStream_t raw_non_default = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&raw_non_default, cudaStreamNonBlocking));
    Stream non_default_stream;
    non_default_stream.s = raw_non_default;
    non_default_stream.owns_ = false;
    EXPECT_NO_THROW((void)classification_correct_count(logits_f32, labels_i32, &non_default_stream));
    CUDA_CHECK(cudaStreamDestroy(raw_non_default));
}

}  // namespace
