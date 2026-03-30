#include "ops.h"

#include <cfloat>
#include <climits>

namespace {

template <int kBlockSize>
__global__ void classification_correct_count_kernel(const float* __restrict__ logits,
                                                    const int32_t* __restrict__ labels,
                                                    int32_t* __restrict__ correct,
                                                    int64_t m,
                                                    int64_t n) {
    __shared__ float s_vals[kBlockSize];
    __shared__ int s_indices[kBlockSize];

    int local_correct = 0;
    for (int64_t row = blockIdx.x; row < m; row += gridDim.x) {
        const int tid = threadIdx.x;
        const int64_t row_offset = row * n;

        float thread_best_val = -FLT_MAX;
        int thread_best_idx = INT_MAX;
        for (int64_t col = tid; col < n; col += kBlockSize) {
            const float v = logits[row_offset + col];
            const int col_i = static_cast<int>(col);
            if (v > thread_best_val || (v == thread_best_val && col_i < thread_best_idx)) {
                thread_best_val = v;
                thread_best_idx = col_i;
            }
        }

        s_vals[tid] = thread_best_val;
        s_indices[tid] = thread_best_idx;
        __syncthreads();

        for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float other_val = s_vals[tid + stride];
                const int other_idx = s_indices[tid + stride];
                if (other_val > s_vals[tid] || (other_val == s_vals[tid] && other_idx < s_indices[tid])) {
                    s_vals[tid] = other_val;
                    s_indices[tid] = other_idx;
                }
            }
            __syncthreads();
        }

        if (tid == 0 && s_indices[0] == labels[row]) {
            ++local_correct;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && local_correct > 0) {
        atomicAdd(correct, local_correct);
    }
}

}  // namespace

Tensor classification_correct_count(const Tensor& logits, const Tensor& labels, Stream& stream) {
    assert_non_default_stream(stream.s, "ops_classification.cu: classification_correct_count");
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

    Tensor correct = Tensor::zeros({1}, DType::I32, Device::CUDA, stream);
    constexpr int kBlockSize = 256;
    const int grid = (m < 1024) ? static_cast<int>(m) : 1024;
    classification_correct_count_kernel<kBlockSize><<<grid, kBlockSize, 0, stream.s>>>(
        static_cast<const float*>(logits.data_),
        static_cast<const int32_t*>(labels.data_),
        static_cast<int32_t*>(correct.data_),
        m,
        n);
    return correct;
}
