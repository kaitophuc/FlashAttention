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


Tensor softmax_forward(const Tensor& X, Stream& stream) {
    assert_non_default_stream(stream.s, "ops_softmax.cu: Softmax_forward");
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

    Tensor Y = Tensor::empty(X.shape_, X.dtype_, X.device_, stream);

    constexpr int BLOCK_SIZE = 256;
    softmax_forward_kernel<BLOCK_SIZE><<<m, BLOCK_SIZE, 0, stream.s>>>(static_cast<const float*>(X.data_), static_cast<float*>(Y.data_), m, n);

    return std::move(Y);
}

SoftmaxGrads softmax_backward(const Tensor& dY, const Tensor& Y, Stream& stream) {
    assert_non_default_stream(stream.s, "ops_softmax.cu: Softmax_backward");
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

    Tensor dX = Tensor::empty(Y.shape_, Y.dtype_, Y.device_, stream);
    constexpr int BLOCK_SIZE = 256;
    softmax_backward_kernel<BLOCK_SIZE><<<m, BLOCK_SIZE, 0, stream.s>>>(static_cast<const float*>(dY.data_), static_cast<const float*>(Y.data_), static_cast<float*>(dX.data_), m, n);

    return SoftmaxGrads{std::move(dX)};
}

template <int BLOCK_SIZE>
__global__ void softmax_cross_entropy_forward_kernel(
    const float* __restrict__ logits,
    const int32_t* __restrict__ labels,
    float* __restrict__ probs,
    float* __restrict__ loss,
    int64_t m,
    int64_t n) {
    const int row = blockIdx.x;
    if (row >= m) return;

    const int64_t row_offset = static_cast<int64_t>(row) * n;

    float thread_max = -FLT_MAX;
    for (int col = threadIdx.x; col < n; col += BLOCK_SIZE) {
        thread_max = fmaxf(thread_max, logits[row_offset + col]);
    }
    const float max_val = blockReduceMax<BLOCK_SIZE>(thread_max);

    float thread_sum = 0.0f;
    for (int col = threadIdx.x; col < n; col += BLOCK_SIZE) {
        const float ex = expf(logits[row_offset + col] - max_val);
        probs[row_offset + col] = ex;
        thread_sum += ex;
    }
    const float sum_val = blockReduceSum<BLOCK_SIZE>(thread_sum);
    const float inv_sum = 1.0f / sum_val;

    for (int col = threadIdx.x; col < n; col += BLOCK_SIZE) {
        probs[row_offset + col] *= inv_sum;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        const int label = labels[row];
        if (label >= 0 && label < n) {
            const float p = fmaxf(probs[row_offset + label], 1e-12f);
            atomicAdd(loss, -logf(p) / static_cast<float>(m));
        }
    }
}

template <int BLOCK_SIZE>
__global__ void softmax_cross_entropy_backward_kernel(
    const float* __restrict__ probs,
    const int32_t* __restrict__ labels,
    float* __restrict__ dX,
    int64_t m,
    int64_t n) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * BLOCK_SIZE + threadIdx.x;
    const int64_t total = m * n;
    if (idx >= total) return;

    const int64_t row = idx / n;
    const int col = static_cast<int>(idx % n);
    const int label = labels[row];

    float grad = probs[idx];
    if (label >= 0 && label < n && col == label) {
        grad -= 1.0f;
    }
    dX[idx] = grad / static_cast<float>(m);
}

SoftmaxCrossEntropyResults softmax_cross_entropy_forward(const Tensor& logits, const Tensor& labels, Stream& stream) {
    assert_non_default_stream(stream.s, "ops_softmax.cu: SoftmaxCrossEntropy_forward");
    if (logits.dtype_ != DType::F32) {
        throw std::invalid_argument("ops_softmax.cu: SoftmaxCrossEntropy_forward: logits must be float32.");
    }
    if (labels.dtype_ != DType::I32) {
        throw std::invalid_argument("ops_softmax.cu: SoftmaxCrossEntropy_forward: labels must be int32.");
    }
    if (logits.device_ != Device::CUDA || labels.device_ != Device::CUDA) {
        throw std::invalid_argument("ops_softmax.cu: SoftmaxCrossEntropy_forward: Only CUDA tensors are supported.");
    }
    if (logits.shape_.size() != 2) {
        throw std::invalid_argument("ops_softmax.cu: SoftmaxCrossEntropy_forward: logits must be a 2D tensor.");
    }
    if (labels.shape_.size() != 1) {
        throw std::invalid_argument("ops_softmax.cu: SoftmaxCrossEntropy_forward: labels must be a 1D tensor.");
    }

    const int64_t m = logits.shape_[0];
    const int64_t n = logits.shape_[1];
    if (m <= 0 || n <= 0) {
        throw std::invalid_argument("ops_softmax.cu: SoftmaxCrossEntropy_forward: logits dimensions must be greater than zero.");
    }
    if (labels.shape_[0] != m) {
        throw std::invalid_argument("ops_softmax.cu: SoftmaxCrossEntropy_forward: labels length must match logits batch dimension.");
    }

    Tensor probs = Tensor::empty(logits.shape_, logits.dtype_, logits.device_, stream);
    Tensor loss = Tensor::zeros({1}, logits.dtype_, logits.device_, stream);

    constexpr int BLOCK_SIZE = 256;
    softmax_cross_entropy_forward_kernel<BLOCK_SIZE><<<m, BLOCK_SIZE, 0, stream.s>>>(
        static_cast<const float*>(logits.data_),
        static_cast<const int32_t*>(labels.data_),
        static_cast<float*>(probs.data_),
        static_cast<float*>(loss.data_),
        m,
        n);

    return SoftmaxCrossEntropyResults{
        std::move(loss),
        SoftmaxCrossEntropyCtx{&labels, std::move(probs), m, n}
    };
}

SoftmaxCrossEntropyGrads softmax_cross_entropy_backward(const SoftmaxCrossEntropyCtx& ctx, Stream& stream) {
    assert_non_default_stream(stream.s, "ops_softmax.cu: SoftmaxCrossEntropy_backward");
    if (ctx.labels == nullptr) {
        throw std::invalid_argument("ops_softmax.cu: SoftmaxCrossEntropy_backward: ctx.labels cannot be null.");
    }

    const Tensor& labels = *ctx.labels;
    const Tensor& probs = ctx.probs;

    if (probs.dtype_ != DType::F32) {
        throw std::invalid_argument("ops_softmax.cu: SoftmaxCrossEntropy_backward: probs must be float32.");
    }
    if (labels.dtype_ != DType::I32) {
        throw std::invalid_argument("ops_softmax.cu: SoftmaxCrossEntropy_backward: labels must be int32.");
    }
    if (probs.device_ != Device::CUDA || labels.device_ != Device::CUDA) {
        throw std::invalid_argument("ops_softmax.cu: SoftmaxCrossEntropy_backward: Only CUDA tensors are supported.");
    }
    if (probs.shape_.size() != 2) {
        throw std::invalid_argument("ops_softmax.cu: SoftmaxCrossEntropy_backward: probs must be a 2D tensor.");
    }
    if (labels.shape_.size() != 1) {
        throw std::invalid_argument("ops_softmax.cu: SoftmaxCrossEntropy_backward: labels must be a 1D tensor.");
    }
    if (ctx.m <= 0 || ctx.n <= 0) {
        throw std::invalid_argument("ops_softmax.cu: SoftmaxCrossEntropy_backward: invalid context dimensions.");
    }
    if (probs.shape_[0] != ctx.m || probs.shape_[1] != ctx.n) {
        throw std::invalid_argument("ops_softmax.cu: SoftmaxCrossEntropy_backward: probs shape must match context dimensions.");
    }
    if (labels.shape_[0] != ctx.m) {
        throw std::invalid_argument("ops_softmax.cu: SoftmaxCrossEntropy_backward: labels length must match context batch dimension.");
    }

    Tensor dX = Tensor::empty(probs.shape_, probs.dtype_, probs.device_, stream);

    constexpr int BLOCK_SIZE = 256;
    const int64_t total = ctx.m * ctx.n;
    const int grid = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
    softmax_cross_entropy_backward_kernel<BLOCK_SIZE><<<grid, BLOCK_SIZE, 0, stream.s>>>(
        static_cast<const float*>(probs.data_),
        static_cast<const int32_t*>(labels.data_),
        static_cast<float*>(dX.data_),
        ctx.m,
        ctx.n);

    return SoftmaxCrossEntropyGrads{std::move(dX)};
}
