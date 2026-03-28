#include "ops.h"
#include <cub/cub.cuh>
#include <cfloat>

struct MaxOp {
    __device__ __forceinline__ float operator()(const float& a, const float& b) const {
        return fmaxf(a, b);
    }
};

template <int BLOCK_SIZE>
__inline__ __device__ float blockReduceMax(float val) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float block_max;
    const float reduced = BlockReduce(temp_storage).Reduce(val, MaxOp{});
    if (threadIdx.x == 0) {
        block_max = reduced;
    }
    __syncthreads();
    return block_max;
}

template <int BLOCK_SIZE>
__inline__ __device__ float blockReduceSum(float val) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float block_sum;
    const float reduced = BlockReduce(temp_storage).Sum(val);
    if (threadIdx.x == 0) {
        block_sum = reduced;
    }
    __syncthreads();
    return block_sum;
}

template <int BLOCK_SIZE>
__global__ void softmax_forward_kernel(const float* __restrict__ X, float* __restrict__ Y, int64_t m, int64_t n) {
    int row = blockIdx.x;
    if (row >= m) return;

    const int64_t row_offset = static_cast<int64_t>(row) * n;
    // Step 1: Find the max value in the row for numerical stability.
    float thread_max = -FLT_MAX;
    for (int col = threadIdx.x; col < n; col += BLOCK_SIZE) {
        thread_max = fmaxf(thread_max, X[row_offset + col]);
    }
    const float max_val = blockReduceMax<BLOCK_SIZE>(thread_max);

    // Step 2: Compute the exponentials and their sum.
    float thread_sum = 0.0f;
    for (int col = threadIdx.x; col < n; col += BLOCK_SIZE) {
        const float exp_val = expf(X[row_offset + col] - max_val);
        Y[row_offset + col] = exp_val; // Store the exponentials temporarily in Y
        thread_sum += exp_val;
    }
    const float sum_val = blockReduceSum<BLOCK_SIZE>(thread_sum); // Using sum reduction to get the sum across threads
    const float inv_sum = 1.0f / sum_val;

    // Step 3: Normalize the exponentials to get the softmax output.
    for (int col = threadIdx.x; col < n; col += BLOCK_SIZE) {
        Y[row_offset + col] *= inv_sum;
    }
}

template <int BLOCK_SIZE>
__global__ void softmax_backward_kernel(const float* __restrict__ dY, const float* __restrict__ Y, float* __restrict__ dX, int64_t m, int64_t n) {
    int row = blockIdx.x;
    if (row >= m) return;   

    const int64_t row_offset = static_cast<int64_t>(row) * n;
    // Step 1: Compute the dot product of Y and dY for the current row
    float thread_dot = 0.0f;
    for (int col = threadIdx.x; col < n; col += BLOCK_SIZE) {
        thread_dot += Y[row_offset + col] * dY[row_offset + col];
    }
    const float dot_val = blockReduceSum<BLOCK_SIZE>(thread_dot); // Using sum reduction to get the dot product across threads

    // Step 2: Compute the gradient w.r.t. the input X
    for (int col = threadIdx.x; col < n; col += BLOCK_SIZE) {
        dX[row_offset + col] = Y[row_offset + col] * (dY[row_offset + col] - dot_val);
    }
}


Tensor softmax_forward(const Tensor& X, Stream* stream) {
    if (stream == nullptr) {
        throw std::invalid_argument("ops_softmax.cu: Softmax_forward: Stream pointer cannot be null.");
    }
    if (stream->s != cudaStream_t(0)) {
        throw std::invalid_argument("ops_softmax.cu: Softmax_forward: Only the default stream is supported at this phase.");
    }
    if (X.dtype_ != DType::F32) {
        throw std::invalid_argument("ops_softmax.cu: Softmax_forward: Only float32 tensors are supported.");
    }
    if (X.shape_.size() != 2) {
        throw std::invalid_argument("ops_softmax.cu: Softmax_forward: X must be a 2D tensor.");
    }

    const int64_t m = X.shape_[0];
    const int64_t n = X.shape_[1];
    if (m <= 0 || n <= 0) {
        throw std::invalid_argument("ops_softmax.cu: Softmax_forward: Input dimensions must be greater than zero.");
    }

    Tensor Y = Tensor::empty(X.shape_, X.dtype_, X.device_, *stream);

    constexpr int BLOCK_SIZE = 256;
    softmax_forward_kernel<BLOCK_SIZE><<<m, BLOCK_SIZE, 0, stream->s>>>(static_cast<const float*>(X.data_), static_cast<float*>(Y.data_), m, n);

    return std::move(Y);
}

SoftmaxGrads softmax_backward(const Tensor& dY, const Tensor& Y, Stream* stream) {
    if (stream == nullptr) {
        throw std::invalid_argument("ops_softmax.cu: Softmax_backward: Stream pointer cannot be null.");
    }
    if (stream->s != cudaStream_t(0)) {
        throw std::invalid_argument("ops_softmax.cu: Softmax_backward: Only the default stream is supported at this phase.");
    }
    if (dY.dtype_ != DType::F32) {
        throw std::invalid_argument("ops_softmax.cu: Softmax_backward: Only float32 tensors are supported.");
    }
    if (dY.shape_ != Y.shape_) {
        throw std::invalid_argument("ops_softmax.cu: Softmax_backward: dY shape must match Y shape from the forward pass.");
    }

    if (Y.dtype_ != DType::F32) {
        throw std::invalid_argument("ops_softmax.cu: Softmax_backward: Only float32 tensors are supported.");
    }
    if (Y.device_ != dY.device_) {
        throw std::invalid_argument("ops_softmax.cu: Softmax_backward: Y and dY must be on the same device.");
    }
    if (Y.shape_ != dY.shape_) {
        throw std::invalid_argument("ops_softmax.cu: Softmax_backward: Y and dY shapes must match.");
    }
    if (Y.shape_.size() != 2) {
        throw std::invalid_argument("ops_softmax.cu: Softmax_backward: Y must be a 2D tensor.");
    }
    const int64_t m = Y.shape_[0];
    const int64_t n = Y.shape_[1];
    if (m <= 0 || n <= 0) {
        throw std::invalid_argument("ops_softmax.cu: Softmax_backward: Y dimensions must be greater than zero.");
    }

    Tensor dX = Tensor::empty(Y.shape_, Y.dtype_, Y.device_, *stream);
    constexpr int BLOCK_SIZE = 256;
    softmax_backward_kernel<BLOCK_SIZE><<<m, BLOCK_SIZE, 0, stream->s>>>(static_cast<const float*>(dY.data_), static_cast<const float*>(Y.data_), static_cast<float*>(dX.data_), m, n);

    return SoftmaxGrads{std::move(dX)};
}
