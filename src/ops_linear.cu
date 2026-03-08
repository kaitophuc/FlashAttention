#include "ops.h"

#include <algorithm>

__global__
void add_bias(float* __restrict__ Y, const float* __restrict__ b, int m, int n) {
    const int total_elements = m * n;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int col = idx % n;
        Y[idx] += b[col];
    }
}

LinearResults linear_forward(const Tensor& X, const Tensor& W, const Tensor* b, Stream* stream, CublasHandle& cublas_handle) {
    // Check input shapes and dtypes.
    // At this phase, assume all tensors are in Float32 for simplicity. In a full implementation, we would handle different dtypes and possibly mixed precision.
    // currently disable CPU support, so we can assume all tensors are on CUDA device.
    if (X.dtype_ != DType::F32 || W.dtype_ != DType::F32) {
        throw std::invalid_argument("Currently only Float32 dtype is supported for X and W.");
    }
    if (X.shape_.size() != 2 || W.shape_.size() != 2) {
        throw std::invalid_argument("X and W must be 2D tensors.");
    }
    if (X.dtype_ != W.dtype_) {
        throw std::invalid_argument("X and W must have the same dtype.");
    }
    if (b != nullptr) {
        if (b->shape_.size() != 1 || b->shape_[0] != W.shape_[0]) {
            throw std::invalid_argument("Bias b must be a 1D tensor with shape matching output features.");
        }
        if (b->dtype_ != X.dtype_) {
            throw std::invalid_argument("Bias b must have the same dtype as X and W.");
        }
    }
    if (X.device_ != W.device_) {
        throw std::invalid_argument("X and W must be on the same device.");
    }
    if (b != nullptr && b->device_ != X.device_) {
        throw std::invalid_argument("Bias b must be on the same device as X and W.");
    }
    if (X.shape_[1] != W.shape_[1]) {
        throw std::invalid_argument("Inner dimensions of X and W must match.");
    }
    if (X.device_ != Device::CUDA || W.device_ != Device::CUDA) {
        throw std::invalid_argument("Currently only CUDA device is supported.");
    }

    // Create output tensor Y.
    std::vector<int64_t> y_shape = {X.shape_[0], W.shape_[0]};
    Tensor Y = Tensor::empty(y_shape, X.dtype_, X.device_, *stream);
    
    // Perform matrix multiplication Y = X * W^T.
    cublasHandle_t handle = cublas_handle.get();
    cublasSetStream(handle, stream->s);
    // Row-major Y(m x n) = X(m x k) * W(n x k)^T mapped to column-major:
    // C_col(n x m) = A_col(k x n)^T * B_col(k x m)
    cublasOperation_t opA = CUBLAS_OP_T;
    cublasOperation_t opB = CUBLAS_OP_N;
    int m = static_cast<int>(X.shape_[0]);
    int n = static_cast<int>(W.shape_[0]);
    int k = static_cast<int>(X.shape_[1]);

    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle, opA, opB, n, m, k,
                &alpha,
                static_cast<float*>(W.data_), k,
                static_cast<float*>(X.data_), k,
                &beta,
                static_cast<float*>(Y.data_), n);

    if (b != nullptr) {
        const int m = static_cast<int>(Y.shape_[0]);
        const int n = static_cast<int>(Y.shape_[1]);
        const int total_elements = m * n;
        
        int min_grid = 0, block = 0;
        cudaOccupancyMaxPotentialBlockSize(&min_grid, &block, add_bias, 0, 0);
        
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, dev);

        int grid = (total_elements + block - 1) / block;
        grid = std::min(grid, prop.multiProcessorCount * 4);

        add_bias<<<grid, block, 0, stream->s>>>(static_cast<float*>(Y.data_), static_cast<const float*>(b->data_), m, n);
    }
    return LinearResults{std::move(Y), LinearCtx{&X, &W}};
}


