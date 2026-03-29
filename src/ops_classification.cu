#include "ops.h"

#include <cfloat>

namespace {

__global__ void classification_correct_count_kernel(const float* __restrict__ logits,
                                                    const int32_t* __restrict__ labels,
                                                    int32_t* __restrict__ correct,
                                                    int64_t m,
                                                    int64_t n) {
    const int row = blockIdx.x;
    if (row >= m) {
        return;
    }

    const int64_t row_offset = static_cast<int64_t>(row) * n;
    float best_val = -FLT_MAX;
    int best_idx = 0;
    for (int64_t col = 0; col < n; ++col) {
        const float v = logits[row_offset + col];
        if (v > best_val) {
            best_val = v;
            best_idx = static_cast<int>(col);
        }
    }

    if (best_idx == labels[row]) {
        atomicAdd(correct, 1);
    }
}

}  // namespace

Tensor classification_correct_count(const Tensor& logits, const Tensor& labels, Stream* stream) {
    if (stream == nullptr) {
        throw std::invalid_argument("ops_classification.cu: classification_correct_count: Stream pointer cannot be null.");
    }
    if (stream->s != cudaStream_t(0)) {
        throw std::invalid_argument("ops_classification.cu: classification_correct_count: Only the default stream is supported at this phase.");
    }
    if (logits.dtype_ != DType::F32) {
        throw std::invalid_argument("ops_classification.cu: classification_correct_count: logits must be float32.");
    }
    if (labels.dtype_ != DType::I32) {
        throw std::invalid_argument("ops_classification.cu: classification_correct_count: labels must be int32.");
    }
    if (logits.device_ != Device::CUDA || labels.device_ != Device::CUDA) {
        throw std::invalid_argument("ops_classification.cu: classification_correct_count: Only CUDA tensors are supported.");
    }
    if (logits.shape_.size() != 2) {
        throw std::invalid_argument("ops_classification.cu: classification_correct_count: logits must be a 2D tensor.");
    }
    if (labels.shape_.size() != 1) {
        throw std::invalid_argument("ops_classification.cu: classification_correct_count: labels must be a 1D tensor.");
    }

    const int64_t m = logits.shape_[0];
    const int64_t n = logits.shape_[1];
    if (m <= 0 || n <= 0) {
        throw std::invalid_argument("ops_classification.cu: classification_correct_count: logits dimensions must be greater than zero.");
    }
    if (labels.shape_[0] != m) {
        throw std::invalid_argument("ops_classification.cu: classification_correct_count: labels length must match logits batch dimension.");
    }

    Tensor correct = Tensor::zeros({1}, DType::I32, Device::CUDA, *stream);
    classification_correct_count_kernel<<<static_cast<int>(m), 1, 0, stream->s>>>(
        static_cast<const float*>(logits.data_),
        static_cast<const int32_t*>(labels.data_),
        static_cast<int32_t*>(correct.data_),
        m,
        n);
    return correct;
}
