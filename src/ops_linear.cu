#include "ops.h"

#include <algorithm>
#include <cstddef>

__global__
void add_bias(float* __restrict__ Y, const float* __restrict__ b, int m, int n) {
    const int total_elements = m * n;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int col = idx % n;
        Y[idx] += b[col];
    }
}

template <int BLOCK_X, int BLOCK_Y, int ROWS_PER_THREAD>
__global__ void compute_db_kernel(const float* __restrict__ dY,
    float* __restrict__ db,
    int M,
    int N) {
    const int col = blockIdx.y * BLOCK_X + threadIdx.x;
    if (col >= N) return;

    const int row_tile_start = blockIdx.x * (BLOCK_Y * ROWS_PER_THREAD);
    const int row_tile_end = min(M, row_tile_start + BLOCK_Y * ROWS_PER_THREAD);

    float sum = 0.0f;
    // Each (threadIdx.x, threadIdx.y) walks rows in its lane.
    for (int row = row_tile_start + threadIdx.y; row < row_tile_end; row += BLOCK_Y) {
        sum += dY[static_cast<size_t>(row) * N + col];
    }

    // Reduce across threadIdx.y for each fixed threadIdx.x (column).
    __shared__ float smem[BLOCK_Y][BLOCK_X];
    smem[threadIdx.y][threadIdx.x] = sum;
    __syncthreads();

    for (int stride = BLOCK_Y / 2; stride > 0; stride >>= 1) {
        if (threadIdx.y < stride) {
            smem[threadIdx.y][threadIdx.x] += smem[threadIdx.y + stride][threadIdx.x];
        }
        __syncthreads();
    }

    // One writer per column per block.
    if (threadIdx.y == 0) {
        atomicAdd(&db[col], smem[0][threadIdx.x]);
    }
}


LinearResults linear_forward(const Tensor& X, const Tensor& W, const Tensor* b, Stream* stream, CublasHandle& handle) {
    // Check input shapes and dtypes.
    // At this phase, assume all tensors are in Float32 for simplicity. In a full implementation, we would handle different dtypes and possibly mixed precision.
    // currently disable CPU support, so we can assume all tensors are on CUDA device.
    if (stream == nullptr) {
        throw std::invalid_argument("stream must not be null.");
    }
    if (stream->s != cudaStream_t(0)) {
        throw std::invalid_argument("Stream argument should be the default stream at this phase.");
    }
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
    // Row-major Y(m x n) = X(m x k) * W(n x k)^T mapped to column-major:
    // Y_col(n x m) = W_col(k x n)^T * X_col(k x m)
    cublasOperation_t opA = CUBLAS_OP_T;
    cublasOperation_t opB = CUBLAS_OP_N;
    int m_out = static_cast<int>(W.shape_[0]);
    int n_out = static_cast<int>(X.shape_[0]);
    int k_out = static_cast<int>(X.shape_[1]);
    int la = static_cast<int>(W.shape_[1]);
    int lb = static_cast<int>(X.shape_[1]);
    int lc = static_cast<int>(W.shape_[0]);
    float alpha = 1.0f;
    float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle.get(), opA, opB, m_out, n_out, k_out,
                                &alpha,
                                static_cast<float*>(W.data_), la,
                                static_cast<float*>(X.data_), lb,
                                &beta,
                                static_cast<float*>(Y.data_), lc));

    if (b != nullptr) {
        const int m = static_cast<int>(Y.shape_[0]);
        const int n = static_cast<int>(Y.shape_[1]);
        const int total_elements = m * n;
        
        int min_grid = 0, block = 0;
        cudaOccupancyMaxPotentialBlockSize(&min_grid, &block, add_bias, 0, 0);

        int grid = (total_elements + block - 1) / block;

        add_bias<<<grid, block, 0, stream->s>>>(static_cast<float*>(Y.data_), static_cast<const float*>(b->data_), m, n);
    }
    return LinearResults{std::move(Y), LinearCtx{&X, &W, b != nullptr, X.shape_[0], W.shape_[0], X.shape_[1]}};
}

LinearGrads linear_backward(const Tensor& dY, const LinearCtx& ctx, bool needs_dX, bool needs_dW, bool needs_db, Stream* stream, CublasHandle& handle) {
    // Check input shapes and dtypes.
    if (stream == nullptr) {
        throw std::invalid_argument("stream must not be null.");
    }
    if (stream->s != cudaStream_t(0)) {
        throw std::invalid_argument("Stream argument should be the default stream at this phase.");
    }
    if (ctx.X == nullptr || ctx.W == nullptr) {
        throw std::invalid_argument("LinearCtx contains null tensor pointers.");
    }
    if (dY.dtype_ != DType::F32) {
        throw std::invalid_argument("Currently only Float32 dtype is supported for dY.");
    }
    if (dY.shape_.size() != 2) {
        throw std::invalid_argument("dY must be a 2D tensor.");
    }
    if (dY.dtype_ != ctx.X->dtype_ || dY.dtype_ != ctx.W->dtype_) {
        throw std::invalid_argument("dY must have the same dtype as X and W.");
    }
    if (dY.device_ != ctx.X->device_ || dY.device_ != ctx.W->device_) {
        throw std::invalid_argument("dY must be on the same device as X and W.");
    }
    if (dY.shape_[0] != ctx.m || dY.shape_[1] != ctx.n) {
        throw std::invalid_argument("Shape of dY must match output shape of forward pass.");
    }
    if (ctx.has_bias && needs_db && (ctx.n <= 0)) {
        throw std::invalid_argument("Invalid shape for bias gradient.");
    }
    if (ctx.k <= 0) {
        throw std::invalid_argument("Invalid inner dimension k in context.");
    }

    std::optional<Tensor> dX;
    if (needs_dX) {
        dX = Tensor::empty(ctx.X->shape_, ctx.X->dtype_, ctx.X->device_, *stream);
    }
    std::optional<Tensor> dW;
    if (needs_dW) {
        dW = Tensor::empty(ctx.W->shape_, ctx.W->dtype_, ctx.W->device_, *stream);
    }
    std::optional<Tensor> db;
    if (ctx.has_bias && needs_db) {
        db = Tensor::empty({ctx.n}, ctx.X->dtype_, ctx.X->device_, *stream);
    }

    // Compute dX = dY * W
    if (needs_dX) {
        // Row-major dX(m x k) = dY(m x n) * W(n x k) mapped to column-major:
        // dX_col(k x m) = W_col(k x n) * dY_col(n x m)
        cublasOperation_t opA = CUBLAS_OP_N;
        cublasOperation_t opB = CUBLAS_OP_N;
        int m_out = static_cast<int>(ctx.W->shape_[1]);
        int n_out = static_cast<int>(dY.shape_[0]);
        int k_out = static_cast<int>(dY.shape_[1]);
        int la = static_cast<int>(ctx.W->shape_[1]);
        int lb = static_cast<int>(dY.shape_[1]);
        int lc = static_cast<int>(ctx.W->shape_[1]);

        float alpha = 1.0f;
        float beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(handle.get(), opA, opB, m_out, n_out, k_out,
                                    &alpha,
                                    static_cast<const float*>(ctx.W->data_), la,
                                    static_cast<const float*>(dY.data_), lb,
                                    &beta,
                                    static_cast<float*>(dX.value().data_), lc));
    }
    
    // Compute dW = dY^T * X
    if (needs_dW) {
        // Row-major dW(n x k) = dY(m x n)^T * X(m x k) mapped to column-major:
        // dW_col(k x n) = X_col(k x m) * dY_col(n x m)^T
        cublasOperation_t opA = CUBLAS_OP_N;
        cublasOperation_t opB = CUBLAS_OP_T;
        int m_out = static_cast<int>(ctx.X->shape_[1]);
        int n_out = static_cast<int>(dY.shape_[1]);
        int k_out = static_cast<int>(ctx.X->shape_[0]);
        int la = static_cast<int>(ctx.X->shape_[1]);
        int lb = static_cast<int>(dY.shape_[1]);
        int lc = static_cast<int>(ctx.X->shape_[1]);
        float alpha = 1.0f;
        float beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(handle.get(), opA, opB, m_out, n_out, k_out,
                                    &alpha,
                                    static_cast<const float*>(ctx.X->data_), la,
                                    static_cast<const float*>(dY.data_), lb,
                                    &beta,
                                    static_cast<float*>(dW.value().data_), lc));
    }

    // Compute db = sum(dY, dim=0)
    if (ctx.has_bias && needs_db) {
        constexpr int BLOCK_X = 128;
        constexpr int BLOCK_Y = 8;
        constexpr int ROWS_PER_THREAD = 4;

        const int m = static_cast<int>(dY.shape_[0]);
        const int n = static_cast<int>(dY.shape_[1]);

        dim3 block(BLOCK_X, BLOCK_Y);
        const int rows_per_block = BLOCK_Y * ROWS_PER_THREAD;
        int grid_x = (m + rows_per_block - 1) / rows_per_block;
        int grid_y = (n + BLOCK_X - 1) / BLOCK_X;
        dim3 grid(grid_x, grid_y);

        db.value().zero_(*stream); // Ensure db is zeroed before accumulation

        compute_db_kernel<BLOCK_X, BLOCK_Y, ROWS_PER_THREAD><<<grid, block, 0, stream->s>>>(
            static_cast<const float*>(dY.data_),
            static_cast<float*>(db.value().data_),
            m,
            n);
    } 
        
    return LinearGrads{
        std::move(dX),
        std::move(dW),
        std::move(db),
        needs_dX,
        needs_dW,
        (ctx.has_bias && needs_db)};
}
